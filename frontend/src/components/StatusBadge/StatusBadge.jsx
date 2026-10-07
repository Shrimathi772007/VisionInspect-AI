import { Badge } from "../Badge/Badge";
import { statusHint, statusLabel, statusTone } from "../../utils/badgeMaps";

/** The ground-truth status badge used on every page: label, tone and tooltip come from badgeMaps. */
export function StatusBadge({ status, fallback }) {
  return (
    <Badge tone={statusTone(status)} title={statusHint(status)}>
      {statusLabel(status) || fallback}
    </Badge>
  );
}
