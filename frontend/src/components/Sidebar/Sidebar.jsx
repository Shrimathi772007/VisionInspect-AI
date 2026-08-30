import { NavLink } from "react-router-dom";
import { LayoutDashboard, Box, ScanEye, UploadCloud, LogOut, ChevronsLeft, ChevronsRight, ScanLine, Database } from "lucide-react";
import { Badge } from "../Badge/Badge";
import { ThemeToggle } from "../ThemeToggle/ThemeToggle";
import { roleLabel, roleTone } from "../../utils/badgeMaps";
import styles from "./Sidebar.module.css";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard, roles: null },
  { to: "/products", label: "Products", icon: Box, roles: null },
  { to: "/inspections", label: "Inspections", icon: ScanEye, roles: null },
  { to: "/inspections/upload", label: "Upload Inspection", icon: UploadCloud, roles: ["quality_engineer"] },
  { to: "/dataset", label: "Dataset Browser", icon: Database, roles: null },
];

export function Sidebar({ collapsed, onToggleCollapsed, mobileOpen, onCloseMobile, user, onLogout }) {
  const visibleItems = NAV_ITEMS.filter((item) => !item.roles || item.roles.includes(user?.role));

  return (
    <>
      {mobileOpen && <div className={styles.scrim} onClick={onCloseMobile} aria-hidden="true" />}
      <aside
        className={`${styles.sidebar} ${collapsed ? styles.collapsed : ""} ${mobileOpen ? styles.mobileOpen : ""}`}
      >
        <div className={styles.brand}>
          <div className={styles.brandMark}>
            <ScanLine size={18} strokeWidth={2} aria-hidden="true" />
          </div>
          {!collapsed && (
            <div className={styles.brandText}>
              <span className={styles.brandName}>VisionInspect</span>
              <span className={styles.brandSub}>AI Quality Platform</span>
            </div>
          )}
        </div>

        <nav className={styles.nav} aria-label="Primary navigation">
          {visibleItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/dashboard" || to === "/inspections"}
              onClick={onCloseMobile}
              className={({ isActive }) => `${styles.navItem} ${isActive ? styles.navItemActive : ""}`}
              title={collapsed ? label : undefined}
            >
              <Icon size={19} strokeWidth={1.9} aria-hidden="true" />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className={styles.footer}>
          {!collapsed && user && (
            <div className={styles.userCard}>
              <div className={styles.avatar}>{user.name?.[0]?.toUpperCase() || "?"}</div>
              <div className={styles.userInfo}>
                <p className={styles.userName} title={user.name}>
                  {user.name}
                </p>
                <Badge tone={roleTone(user.role)}>{roleLabel(user.role)}</Badge>
              </div>
            </div>
          )}
          <div className={`${styles.themeRow} ${collapsed ? styles.collapsed : ""}`}>
            {!collapsed && <span className={styles.themeLabel}>Theme</span>}
            <ThemeToggle compact={collapsed} />
          </div>
          <button type="button" className={styles.logoutButton} onClick={onLogout} title="Log out">
            <LogOut size={18} strokeWidth={1.9} aria-hidden="true" />
            {!collapsed && <span>Log out</span>}
          </button>
          <button
            type="button"
            className={styles.collapseButton}
            onClick={onToggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronsRight size={16} /> : <ChevronsLeft size={16} />}
          </button>
        </div>
      </aside>
    </>
  );
}
