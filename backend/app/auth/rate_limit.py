"""In-memory limit on failed logins, per (client IP, email) and per client IP.

Only failures count, in a sliding window. While a key is over its limit every login attempt for it is refused
with 429 before the password is checked, so the response never tells whether the password was right. A
successful login clears the (IP, email) counter only; the per-IP counter keeps running.

State lives in this process: it is correct with one uvicorn worker only, and a restart clears it. The client IP
is request.client.host, which behind nginx is the address nginx saw (see docs/DEPLOY.md).
"""

import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_MAX_ATTEMPTS_PER_IP = 20
DEFAULT_WINDOW_SECONDS = 15 * 60
MAX_TRACKED_KEYS = 10_000
SWEEP_INTERVAL_SECONDS = 60

TOO_MANY_ATTEMPTS_MESSAGE = "Too many failed login attempts. Please wait and try again later."


def positive_int_from_env(name: str, default: int) -> int:
    """The positive integer in environment variable `name`, or `default` if it is unset or not one."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value < 1:
        logger.warning("%s=%r is not a positive integer; using the default %d", name, raw, default)
        return default
    return value


def normalize_login_email(email: str) -> str:
    return email.strip().lower()


class LoginRateLimiter:
    def __init__(
        self,
        max_attempts: int,
        max_attempts_per_ip: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = MAX_TRACKED_KEYS,
    ) -> None:
        self.max_attempts = max_attempts
        self.max_attempts_per_ip = max_attempts_per_ip
        self.window_seconds = window_seconds
        self.clock = clock
        self.max_keys = max_keys
        # key -> failure timestamps, oldest first, at most the key's limit long (older ones no longer matter).
        self._failures: dict[tuple, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def _keys(self, client_ip: str, email: str) -> list[tuple[tuple, int]]:
        return [
            (("ip_email", client_ip, normalize_login_email(email)), self.max_attempts),
            (("ip", client_ip), self.max_attempts_per_ip),
        ]

    def _prune(self, key: tuple, now: float) -> deque[float] | None:
        failures = self._failures.get(key)
        if failures is None:
            return None
        while failures and failures[0] <= now - self.window_seconds:
            failures.popleft()
        if not failures:
            del self._failures[key]
            return None
        return failures

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        for key in list(self._failures):
            self._prune(key, now)

    def retry_after(self, client_ip: str, email: str) -> int | None:
        """Seconds until this client may try `email` again, or None if it may try now."""
        with self._lock:
            now = self.clock()
            self._sweep(now)
            wait = 0.0
            for key, limit in self._keys(client_ip, email):
                failures = self._prune(key, now)
                if failures is not None and len(failures) >= limit:
                    wait = max(wait, failures[-limit] + self.window_seconds - now)
            return max(1, math.ceil(wait)) if wait > 0 else None

    def record_failure(self, client_ip: str, email: str) -> None:
        with self._lock:
            now = self.clock()
            for key, limit in self._keys(client_ip, email):
                failures = self._failures.pop(key, None)
                if failures is None or failures.maxlen != limit:
                    failures = deque(failures or (), maxlen=limit)
                failures.append(now)
                # Re-inserted, so the dict stays ordered from least to most recently failed.
                self._failures[key] = failures
            while len(self._failures) > self.max_keys:
                del self._failures[next(iter(self._failures))]

    def reset(self, client_ip: str, email: str) -> None:
        """Clear the (IP, email) counter after a successful login. The per-IP counter is kept."""
        with self._lock:
            self._failures.pop(self._keys(client_ip, email)[0][0], None)

    def clear(self) -> None:
        with self._lock:
            self._failures.clear()
            self._last_sweep = self.clock()

    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._failures)


def limiter_from_env() -> LoginRateLimiter:
    return LoginRateLimiter(
        max_attempts=positive_int_from_env("LOGIN_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
        max_attempts_per_ip=positive_int_from_env("LOGIN_MAX_ATTEMPTS_PER_IP", DEFAULT_MAX_ATTEMPTS_PER_IP),
        window_seconds=positive_int_from_env("LOGIN_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS),
    )


login_rate_limiter = limiter_from_env()
