import { useEffect, useRef } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../../auth/useAuth";
import { useToast } from "../Toast/ToastProvider";

/**
 * Route-level counterpart of RoleGate: renders `children` only for the allowed roles and
 * otherwise redirects (with an info toast). Place it inside ProtectedRoute, so the user is
 * already loaded. Like RoleGate this is a UX convenience - the backend's require_role is the
 * actual authorization boundary.
 */
export function RoleRoute({
  allow,
  children,
  redirectTo = "/dashboard",
  message = "Quality Engineer access required",
}) {
  const { user } = useAuth();
  const { showToast } = useToast();
  const isAllowed = Boolean(user) && allow.includes(user.role);
  const hasNotified = useRef(false);

  useEffect(() => {
    if (isAllowed || hasNotified.current) return;
    hasNotified.current = true;
    showToast({ type: "info", title: message });
  }, [isAllowed, message, showToast]);

  if (!isAllowed) return <Navigate to={redirectTo} replace />;
  return children;
}
