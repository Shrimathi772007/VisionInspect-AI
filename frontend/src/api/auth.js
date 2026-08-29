import { apiFetch } from "./client";

export function login(email, password) {
  return apiFetch("/auth/login", { method: "POST", json: { email, password } });
}

export function fetchCurrentUser() {
  return apiFetch("/auth/me");
}
