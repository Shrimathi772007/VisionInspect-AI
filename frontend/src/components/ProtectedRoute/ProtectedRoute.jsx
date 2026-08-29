import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../../auth/useAuth";
import { FullPageSpinner } from "../FullPageSpinner/FullPageSpinner";

export function ProtectedRoute({ children }) {
  const { isAuthenticated, isBootstrapping } = useAuth();
  const location = useLocation();

  if (isBootstrapping) return <FullPageSpinner />;

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}
