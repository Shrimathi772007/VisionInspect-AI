import { createContext, useCallback, useEffect, useMemo, useState } from "react";
import { fetchCurrentUser, login as loginRequest } from "../api/auth";
import { setAuthToken, setUnauthorizedHandler } from "../api/client";

const TOKEN_STORAGE_KEY = "visioninspect.token";

export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isBootstrapping, setIsBootstrapping] = useState(true);

  const logout = useCallback(() => {
    setAuthToken(null);
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setUser(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(logout);
  }, [logout]);

  useEffect(() => {
    const storedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (!storedToken) {
      setIsBootstrapping(false);
      return;
    }

    setAuthToken(storedToken);
    fetchCurrentUser()
      .then(setUser)
      .catch(() => {
        setAuthToken(null);
        localStorage.removeItem(TOKEN_STORAGE_KEY);
      })
      .finally(() => setIsBootstrapping(false));
  }, []);

  const login = useCallback(async (email, password) => {
    const result = await loginRequest(email, password);
    setAuthToken(result.access_token);
    localStorage.setItem(TOKEN_STORAGE_KEY, result.access_token);
    setUser(result.user);
    return result.user;
  }, []);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      isBootstrapping,
      login,
      logout,
    }),
    [user, isBootstrapping, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
