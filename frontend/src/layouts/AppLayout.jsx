import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "../components/Sidebar/Sidebar";
import { Topbar } from "../components/Topbar/Topbar";
import { useAuth } from "../auth/useAuth";
import styles from "./AppLayout.module.css";

const COLLAPSE_STORAGE_KEY = "visioninspect.sidebarCollapsed";

export function AppLayout() {
  const { user, logout } = useAuth();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  });
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_STORAGE_KEY, String(collapsed));
    } catch {
      // localStorage unavailable — non-critical UX preference
    }
  }, [collapsed]);

  return (
    <div className={styles.shell}>
      <Sidebar
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((c) => !c)}
        mobileOpen={mobileNavOpen}
        onCloseMobile={() => setMobileNavOpen(false)}
        user={user}
        onLogout={logout}
      />
      <div className={`${styles.content} ${collapsed ? styles.contentCollapsed : ""}`}>
        <Topbar onOpenMobileNav={() => setMobileNavOpen(true)} />
        <main className={styles.main}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
