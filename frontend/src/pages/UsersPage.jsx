import { useState } from "react";
import { Users } from "lucide-react";
import { useAuth } from "../auth/useAuth";
import { useUsers } from "../hooks/useUsers";
import { ApiError } from "../api/client";
import { useToast } from "../components/Toast/ToastProvider";
import { PageHeader } from "../components/PageHeader/PageHeader";
import { Card } from "../components/Card/Card";
import { Button } from "../components/Button/Button";
import { Select } from "../components/Select/Select";
import { Modal } from "../components/Modal/Modal";
import { EmptyState } from "../components/EmptyState/EmptyState";
import { ErrorState } from "../components/ErrorState/ErrorState";
import { Skeleton } from "../components/Skeleton/Skeleton";
import { Badge } from "../components/Badge/Badge";
import { ROLE_LABELS, roleLabel, roleTone } from "../utils/badgeMaps";
import { formatDateTime } from "../utils/formatDate";
import styles from "./UsersPage.module.css";

const OWN_ROLE_HINT = "You cannot change your own role.";

export function UsersPage() {
  const { user: currentUser } = useAuth();
  const { users, isLoading, error, refetch, changeRole } = useUsers();
  const { showToast } = useToast();

  // { user, role } while the confirmation dialog is open.
  const [pendingChange, setPendingChange] = useState(null);
  const [isSaving, setIsSaving] = useState(false);

  const closeConfirm = () => {
    if (isSaving) return;
    setPendingChange(null);
  };

  const confirmChange = async () => {
    if (!pendingChange) return;
    const { user, role } = pendingChange;
    setIsSaving(true);
    try {
      await changeRole(user.id, role);
      showToast({
        type: "success",
        title: "Role updated",
        message: `${user.name} is now a ${roleLabel(role)}.`,
      });
      setPendingChange(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        showToast({
          type: "error",
          title: "Can't change this role",
          message: "At least one Quality Engineer must remain.",
        });
        setPendingChange(null);
      } else if (err instanceof ApiError && err.status === 404) {
        showToast({ type: "error", title: "User not found", message: "The list has been refreshed." });
        setPendingChange(null);
        refetch();
      } else {
        showToast({
          type: "error",
          title: "Couldn't change role",
          message: err instanceof ApiError ? err.message : "Something went wrong.",
        });
      }
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Administration"
        title="Users"
        description="Review accounts and choose who is a Quality Engineer. New accounts start as Factory Supervisor."
      />

      {isLoading && (
        <Card className={styles.tableCard}>
          <div className={styles.skeletonList} role="status" aria-label="Loading users">
            {[1, 2, 3, 4].map((key) => (
              <div key={key} className={styles.skeletonRow}>
                <Skeleton width="22%" height={14} />
                <Skeleton width="30%" height={14} />
                <Skeleton width="14%" height={20} />
                <Skeleton width="18%" height={14} />
              </div>
            ))}
          </div>
        </Card>
      )}

      {!isLoading && error && (
        <Card className={styles.tableCard}>
          <ErrorState message={`Failed to load users. ${error.message}`} onRetry={refetch} />
        </Card>
      )}

      {!isLoading && !error && users.length === 0 && (
        <Card>
          <EmptyState icon={Users} title="No users yet" description="Accounts appear here once people register." />
        </Card>
      )}

      {!isLoading && !error && users.length > 0 && (
        <Card className={styles.tableCard}>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Name</th>
                  <th scope="col">Email</th>
                  <th scope="col">Role</th>
                  <th scope="col">Joined</th>
                  <th scope="col">Change role</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => {
                  const isSelf = user.id === currentUser?.id;
                  return (
                    <tr key={user.id}>
                      <td className={styles.nameCell}>
                        {user.name}
                        {isSelf && <span className={styles.youTag}>You</span>}
                      </td>
                      <td className={styles.emailCell}>{user.email}</td>
                      <td>
                        <Badge tone={roleTone(user.role)}>{roleLabel(user.role)}</Badge>
                      </td>
                      <td className={styles.dateCell}>{formatDateTime(user.created_at)}</td>
                      <td className={styles.controlCell}>
                        <Select
                          aria-label={`Role for ${user.name}`}
                          value={user.role}
                          disabled={isSelf}
                          title={isSelf ? OWN_ROLE_HINT : undefined}
                          helperText={isSelf ? OWN_ROLE_HINT : undefined}
                          onChange={(event) => setPendingChange({ user, role: event.target.value })}
                        >
                          {Object.entries(ROLE_LABELS).map(([value, label]) => (
                            <option key={value} value={value}>
                              {label}
                            </option>
                          ))}
                        </Select>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Modal
        open={Boolean(pendingChange)}
        onClose={closeConfirm}
        title={pendingChange ? `Change role for ${pendingChange.user.name}?` : ""}
        description={
          pendingChange
            ? `${pendingChange.user.name} will become a ${roleLabel(pendingChange.role)}. This takes effect on their next request.`
            : undefined
        }
        footer={
          <>
            <Button variant="ghost" onClick={closeConfirm} disabled={isSaving}>
              Cancel
            </Button>
            <Button onClick={confirmChange} loading={isSaving}>
              Change Role
            </Button>
          </>
        }
      >
        {pendingChange && (
          <p className={styles.confirmText}>
            {roleLabel(pendingChange.user.role)} &rarr; <strong>{roleLabel(pendingChange.role)}</strong>
          </p>
        )}
      </Modal>
    </div>
  );
}
