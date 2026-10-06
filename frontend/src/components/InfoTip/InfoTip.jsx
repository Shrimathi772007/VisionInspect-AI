import { Info } from "lucide-react";
import styles from "./InfoTip.module.css";

/**
 * A small info icon that explains a term: the text is the hover tooltip (title) and the accessible name,
 * and the icon can be focused with the keyboard. It adds no text nodes, so it can sit inside a label
 * without changing the label's text.
 */
export function InfoTip({ text }) {
  if (!text) return null;
  return (
    <span className={styles.tip} title={text} aria-label={text} role="img" tabIndex={0}>
      <Info size={13} strokeWidth={2} aria-hidden="true" />
    </span>
  );
}
