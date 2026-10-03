import { apiFetch } from "./client";

export function login(email, password) {
  return apiFetch("/auth/login", { method: "POST", json: { email, password } });
}

export function fetchCurrentUser() {
  return apiFetch("/auth/me");
}

// Sends ONLY name, email and password: the server always creates a factory_supervisor and a
// role must never be part of this request.
export function registerUser({ name, email, password }) {
  return apiFetch("/auth/register", { method: "POST", json: { name, email, password } });
}
