"""Failed-login limiting: POST /auth/login answers 429 after too many failures per (IP, email) or per IP."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.auth.rate_limit import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_ATTEMPTS_PER_IP,
    DEFAULT_WINDOW_SECONDS,
    TOO_MANY_ATTEMPTS_MESSAGE,
    LoginRateLimiter,
    limiter_from_env,
    login_rate_limiter,
    positive_int_from_env,
)
from app.main import app
from app.models.user import UserRole

REPO_ROOT = Path(__file__).resolve().parents[2]
WRONG_PASSWORD = "DefinitelyWrong123"


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(login_rate_limiter, "clock", fake)
    login_rate_limiter.clear()
    return fake


@pytest.fixture
def account(temp_users):
    _, email = temp_users.create(UserRole.factory_supervisor)
    return email, temp_users.password


def _login(client, email, password, **kwargs):
    return client.post("/auth/login", json={"email": email, "password": password}, **kwargs)


def _fail(client, email, times, **kwargs):
    for _ in range(times):
        assert _login(client, email, WRONG_PASSWORD, **kwargs).status_code == 401


def test_defaults_are_five_per_email_and_twenty_per_ip_in_fifteen_minutes():
    assert (DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_ATTEMPTS_PER_IP, DEFAULT_WINDOW_SECONDS) == (5, 20, 900)


def test_lockout_after_max_failures_even_with_the_right_password(client, clock, account):
    email, password = account
    _fail(client, email, login_rate_limiter.max_attempts)

    response = _login(client, email, password)
    assert response.status_code == 429
    assert response.json() == {"detail": TOO_MANY_ATTEMPTS_MESSAGE}
    assert response.headers["Retry-After"] == str(login_rate_limiter.window_seconds)


def test_failures_below_the_limit_do_not_lock(client, clock, account):
    email, password = account
    _fail(client, email, login_rate_limiter.max_attempts - 1)
    assert _login(client, email, password).status_code == 200


def test_retry_after_counts_down_with_the_clock(client, clock, account):
    email, _ = account
    _fail(client, email, login_rate_limiter.max_attempts)
    clock.now += 100.5
    response = _login(client, email, WRONG_PASSWORD)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == str(login_rate_limiter.window_seconds - 100)


def test_unknown_email_gets_the_same_429_as_a_real_account(client, clock, account, temp_users):
    email, _ = account
    unknown = temp_users.email()  # never created
    _fail(client, email, login_rate_limiter.max_attempts)
    _fail(client, unknown, login_rate_limiter.max_attempts)

    real, missing = _login(client, email, WRONG_PASSWORD), _login(client, unknown, WRONG_PASSWORD)
    assert real.status_code == missing.status_code == 429
    assert real.json() == missing.json() == {"detail": TOO_MANY_ATTEMPTS_MESSAGE}
    assert real.headers["Retry-After"] == missing.headers["Retry-After"]


def test_success_resets_the_email_counter(client, clock, account):
    email, password = account
    _fail(client, email, login_rate_limiter.max_attempts - 1)
    assert _login(client, email, password).status_code == 200
    # A fresh allowance: another max-1 failures are still not a lockout.
    _fail(client, email, login_rate_limiter.max_attempts - 1)
    assert _login(client, email, password).status_code == 200


def test_email_key_is_case_and_whitespace_insensitive():
    limiter = LoginRateLimiter(max_attempts=2, max_attempts_per_ip=100, window_seconds=60, clock=FakeClock())
    limiter.record_failure("10.0.0.1", "Someone@Example.com")
    limiter.record_failure("10.0.0.1", "  someone@example.COM ")
    assert limiter.retry_after("10.0.0.1", "someone@example.com") is not None


def test_a_different_email_is_unaffected(client, clock, account, temp_users):
    email, _ = account
    _, other_email = temp_users.create(UserRole.factory_supervisor)
    _fail(client, email, login_rate_limiter.max_attempts)
    assert _login(client, email, temp_users.password).status_code == 429
    assert _login(client, other_email, temp_users.password).status_code == 200


def test_a_different_ip_is_unaffected(clock, account):
    email, password = account
    attacker = TestClient(app, client=("203.0.113.10", 50000))
    user = TestClient(app, client=("198.51.100.20", 50000))
    _fail(attacker, email, login_rate_limiter.max_attempts)
    assert _login(attacker, email, password).status_code == 429
    assert _login(user, email, password).status_code == 200


def test_per_ip_limit_across_many_emails(clock, temp_users):
    attacker = TestClient(app, client=("203.0.113.11", 50000))
    other_ip = TestClient(app, client=("198.51.100.21", 50000))
    _, email = temp_users.create(UserRole.factory_supervisor)
    per_email = login_rate_limiter.max_attempts - 1
    for start in range(0, login_rate_limiter.max_attempts_per_ip, per_email):
        _fail(attacker, temp_users.email(), min(per_email, login_rate_limiter.max_attempts_per_ip - start))

    response = _login(attacker, email, temp_users.password)
    assert response.status_code == 429
    assert response.headers["Retry-After"] == str(login_rate_limiter.window_seconds)
    assert _login(other_ip, email, temp_users.password).status_code == 200


def test_success_does_not_reset_the_per_ip_counter(clock, temp_users):
    attacker = TestClient(app, client=("203.0.113.12", 50000))
    _, own_email = temp_users.create(UserRole.factory_supervisor)
    _, target = temp_users.create(UserRole.factory_supervisor)
    per_email = login_rate_limiter.max_attempts - 1
    for start in range(0, login_rate_limiter.max_attempts_per_ip - 1, per_email):
        _fail(attacker, temp_users.email(), min(per_email, login_rate_limiter.max_attempts_per_ip - 1 - start))
    assert _login(attacker, own_email, temp_users.password).status_code == 200
    _fail(attacker, target, 1)
    assert _login(attacker, target, temp_users.password).status_code == 429


def test_lockout_expires_after_the_window(client, clock, account):
    email, password = account
    _fail(client, email, login_rate_limiter.max_attempts)
    clock.now += login_rate_limiter.window_seconds - 1
    assert _login(client, email, password).status_code == 429
    clock.now += 1
    assert _login(client, email, password).status_code == 200


def test_sliding_window_frees_one_attempt_as_the_oldest_failure_expires(client, clock, account):
    email, password = account
    _fail(client, email, 1)
    clock.now += 60
    _fail(client, email, login_rate_limiter.max_attempts - 1)
    assert _login(client, email, password).status_code == 429
    clock.now += login_rate_limiter.window_seconds - 60
    assert _login(client, email, WRONG_PASSWORD).status_code == 401
    assert _login(client, email, password).status_code == 429


def test_registration_is_not_rate_limited(client, clock, account, temp_users):
    email, _ = account
    _fail(client, email, login_rate_limiter.max_attempts)
    response = client.post(
        "/auth/register", json={"name": "Temp Registrant", "email": temp_users.email(), "password": "TempPass123"}
    )
    assert response.status_code == 201


def test_validation_errors_are_not_counted(client, clock, account):
    email, password = account
    for _ in range(login_rate_limiter.max_attempts + 1):
        assert client.post("/auth/login", json={"email": "not-an-email", "password": "x"}).status_code == 422
    assert _login(client, email, password).status_code == 200


# --- Real client IP behind the proxy ----------------------------------------------------------------------------

NGINX_IP = "172.18.0.5"


def _behind_proxy(peer_ip: str) -> TestClient:
    # What uvicorn --proxy-headers --forwarded-allow-ips <NGINX_IP> wraps the app in.
    return TestClient(ProxyHeadersMiddleware(app, trusted_hosts=NGINX_IP), client=(peer_ip, 50000))


def test_spoofed_forwarded_for_from_an_untrusted_client_is_ignored(clock, account):
    email, password = account
    direct = _behind_proxy("203.0.113.13")
    for n in range(login_rate_limiter.max_attempts):
        response = _login(direct, email, WRONG_PASSWORD, headers={"X-Forwarded-For": f"192.0.2.{n}"})
        assert response.status_code == 401
    # A fresh spoofed address does not help: the key is the real peer address.
    response = _login(direct, email, password, headers={"X-Forwarded-For": "192.0.2.200"})
    assert response.status_code == 429


def test_forwarded_for_from_the_trusted_proxy_is_the_client_ip(clock, account):
    email, password = account
    nginx = _behind_proxy(NGINX_IP)
    _fail(nginx, email, login_rate_limiter.max_attempts, headers={"X-Forwarded-For": "198.51.100.30"})
    assert _login(nginx, email, password, headers={"X-Forwarded-For": "198.51.100.30"}).status_code == 429
    # Another real client behind the same nginx is not locked out with them.
    assert _login(nginx, email, password, headers={"X-Forwarded-For": "198.51.100.31"}).status_code == 200


def test_nginx_overwrites_x_forwarded_for_with_the_peer_address():
    conf = (REPO_ROOT / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in conf
    assert "$proxy_add_x_forwarded_for" not in conf


def test_entrypoint_takes_forwarded_allow_ips_from_the_environment():
    entrypoint = (REPO_ROOT / "backend" / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert '--forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}"' in entrypoint
    assert "--proxy-headers" in entrypoint


# --- Limiter internals ------------------------------------------------------------------------------------------


def test_expired_keys_are_purged_by_the_periodic_sweep():
    clock = FakeClock()
    limiter = LoginRateLimiter(max_attempts=5, max_attempts_per_ip=20, window_seconds=60, clock=clock)
    for n in range(50):
        limiter.record_failure(f"10.0.0.{n}", "a@example.com")
    assert limiter.tracked_keys() == 100
    clock.now += 61
    limiter.retry_after("10.9.9.9", "b@example.com")
    assert limiter.tracked_keys() == 0


def test_tracked_keys_are_capped_least_recent_first():
    clock = FakeClock()
    limiter = LoginRateLimiter(max_attempts=5, max_attempts_per_ip=20, window_seconds=60, clock=clock, max_keys=10)
    for n in range(20):
        limiter.record_failure(f"10.0.0.{n}", "a@example.com")
    assert limiter.tracked_keys() == 10
    assert limiter.retry_after("10.0.0.19", "a@example.com") is None  # still tracked, under the limit
    assert ("ip", "10.0.0.19") in limiter._failures
    assert ("ip", "10.0.0.0") not in limiter._failures


def test_reset_keeps_the_per_ip_counter():
    limiter = LoginRateLimiter(max_attempts=5, max_attempts_per_ip=20, window_seconds=60, clock=FakeClock())
    limiter.record_failure("10.0.0.1", "a@example.com")
    limiter.reset("10.0.0.1", "a@example.com")
    assert ("ip_email", "10.0.0.1", "a@example.com") not in limiter._failures
    assert len(limiter._failures[("ip", "10.0.0.1")]) == 1


@pytest.mark.parametrize("raw", ["0", "-3", "abc", "1.5", "  "])
def test_bad_environment_values_fall_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("LOGIN_TEST_LIMIT", raw)
    assert positive_int_from_env("LOGIN_TEST_LIMIT", 7) == 7


def test_valid_environment_values_are_used(monkeypatch):
    monkeypatch.setenv("LOGIN_TEST_LIMIT", " 12 ")
    assert positive_int_from_env("LOGIN_TEST_LIMIT", 7) == 12
    monkeypatch.delenv("LOGIN_TEST_LIMIT")
    assert positive_int_from_env("LOGIN_TEST_LIMIT", 7) == 7


def test_limiter_from_env_reads_the_environment(monkeypatch):
    monkeypatch.setenv("LOGIN_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("LOGIN_MAX_ATTEMPTS_PER_IP", "nope")
    monkeypatch.setenv("LOGIN_WINDOW_SECONDS", "120")
    limiter = limiter_from_env()
    assert (limiter.max_attempts, limiter.max_attempts_per_ip, limiter.window_seconds) == (3, 20, 120)
