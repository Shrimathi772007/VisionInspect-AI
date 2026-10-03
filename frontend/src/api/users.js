import { apiFetch } from "./client";

export function listUsers() {
  return apiFetch("/users");
}

export function updateUserRole(userId, role) {
  return apiFetch(`/users/${userId}/role`, { method: "PATCH", json: { role } });
}
