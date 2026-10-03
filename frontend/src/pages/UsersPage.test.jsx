import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("../api/users", () => ({ listUsers: vi.fn(), updateUserRole: vi.fn() }));
vi.mock("../auth/useAuth", () => ({
  useAuth: () => ({ user: { id: 1, name: "Ada QE", role: "quality_engineer" } }),
}));

import { listUsers, updateUserRole } from "../api/users";
import { ApiError } from "../api/client";
import { ToastProvider } from "../components/Toast/ToastProvider";
import { UsersPage } from "./UsersPage";

const NOW = "2026-10-01T10:00:00Z";
const USERS = [
  { id: 1, name: "Ada QE", email: "ada@example.com", role: "quality_engineer", created_at: NOW },
  { id: 2, name: "Bob Supervisor", email: "bob@example.com", role: "factory_supervisor", created_at: NOW },
  { id: 3, name: "Cy QE", email: "cy@example.com", role: "quality_engineer", created_at: NOW },
];

function renderPage() {
  return render(
    <ToastProvider>
      <UsersPage />
    </ToastProvider>
  );
}

const roleSelect = (name) => screen.getByRole("combobox", { name: `Role for ${name}` });

async function requestChange(name, role) {
  await screen.findByText(name);
  fireEvent.change(roleSelect(name), { target: { value: role } });
  return screen.findByRole("dialog");
}

describe("UsersPage", () => {
  beforeEach(() => {
    listUsers.mockReset().mockResolvedValue(USERS);
    updateUserRole.mockReset();
  });

  it("lists the users returned by the API with role badges", async () => {
    renderPage();
    for (const user of USERS) {
      expect(await screen.findByText(user.name)).toBeInTheDocument();
      expect(screen.getByText(user.email)).toBeInTheDocument();
    }
    const bobRow = screen.getByText("Bob Supervisor").closest("tr");
    expect(within(bobRow).getAllByText("Factory Supervisor").length).toBeGreaterThan(0);
  });

  it("disables the role control for the current user only", async () => {
    renderPage();
    await screen.findByText("Ada QE");
    expect(roleSelect("Ada QE")).toBeDisabled();
    expect(roleSelect("Ada QE")).toHaveAttribute("title", "You cannot change your own role.");
    expect(screen.getByText("You cannot change your own role.")).toBeInTheDocument();
    expect(roleSelect("Bob Supervisor")).not.toBeDisabled();
    expect(roleSelect("Cy QE")).not.toBeDisabled();
  });

  it("asks for confirmation, then updates the role and refetches", async () => {
    updateUserRole.mockResolvedValue({ ...USERS[1], role: "quality_engineer" });
    renderPage();

    const dialog = await requestChange("Bob Supervisor", "quality_engineer");
    expect(within(dialog).getByText("Change role for Bob Supervisor?")).toBeInTheDocument();
    expect(updateUserRole).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole("button", { name: "Change Role" }));

    await waitFor(() => expect(updateUserRole).toHaveBeenCalledWith(2, "quality_engineer"));
    await waitFor(() => expect(listUsers).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("Role updated")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("does nothing when the confirmation is cancelled", async () => {
    renderPage();
    const dialog = await requestChange("Bob Supervisor", "quality_engineer");
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(updateUserRole).not.toHaveBeenCalled();
    expect(roleSelect("Bob Supervisor")).toHaveValue("factory_supervisor");
  });

  it("explains a 409 as the last-Quality-Engineer rule", async () => {
    updateUserRole.mockRejectedValue(new ApiError(409, "Cannot change the role of the last quality engineer"));
    renderPage();

    const dialog = await requestChange("Cy QE", "factory_supervisor");
    fireEvent.click(within(dialog).getByRole("button", { name: "Change Role" }));

    expect(await screen.findByText("At least one Quality Engineer must remain.")).toBeInTheDocument();
    expect(updateUserRole).toHaveBeenCalledWith(3, "factory_supervisor");
    expect(listUsers).toHaveBeenCalledTimes(1);
  });

  it("shows 'User not found' on 404 and refreshes the list", async () => {
    updateUserRole.mockRejectedValue(new ApiError(404, "User not found"));
    renderPage();

    const dialog = await requestChange("Bob Supervisor", "quality_engineer");
    fireEvent.click(within(dialog).getByRole("button", { name: "Change Role" }));

    expect(await screen.findByText("User not found")).toBeInTheDocument();
    await waitFor(() => expect(listUsers).toHaveBeenCalledTimes(2));
  });

  it("renders a loading skeleton while users load", () => {
    listUsers.mockReturnValue(new Promise(() => {}));
    renderPage();
    expect(screen.getByRole("status", { name: "Loading users" })).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("renders an error state with a working retry", async () => {
    listUsers.mockRejectedValueOnce(new ApiError(500, "Server unavailable"));
    renderPage();

    expect(await screen.findByText(/Failed to load users\./)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Bob Supervisor")).toBeInTheDocument();
    expect(listUsers).toHaveBeenCalledTimes(2);
  });

  it("renders an empty state when there are no users", async () => {
    listUsers.mockResolvedValue([]);
    renderPage();
    expect(await screen.findByText("No users yet")).toBeInTheDocument();
  });
});
