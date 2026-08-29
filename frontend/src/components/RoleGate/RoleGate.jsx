import { useAuth } from "../../auth/useAuth";

/**
 * Purely a UX convenience for hiding/showing UI by role.
 * The backend (require_role) is the actual authorization boundary —
 * this component must never be relied on for security.
 */
export function RoleGate({ allow, children, fallback = null }) {
  const { user } = useAuth();
  if (!user || !allow.includes(user.role)) return fallback;
  return children;
}
