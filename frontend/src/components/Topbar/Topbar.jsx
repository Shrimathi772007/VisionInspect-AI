import { Menu, ScanLine } from "lucide-react";
import { ThemeToggle } from "../ThemeToggle/ThemeToggle";
import styles from "./Topbar.module.css";

export function Topbar({ onOpenMobileNav, title }) {
  return (
    <header className={styles.topbar}>
      <button
        type="button"
        className={styles.menuButton}
        onClick={onOpenMobileNav}
        aria-label="Open navigation menu"
      >
        <Menu size={20} />
      </button>
      <div className={styles.brand}>
        <ScanLine size={16} className={styles.brandIcon} aria-hidden="true" />
        <span>{title || "VisionInspect AI"}</span>
      </div>
      <div className={styles.actions}>
        <ThemeToggle compact />
      </div>
    </header>
  );
}
