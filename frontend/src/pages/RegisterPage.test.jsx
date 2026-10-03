import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../api/auth", () => ({ registerUser: vi.fn() }));
vi.mock("../auth/useAuth", () => ({ useAuth: () => ({ isAuthenticated: false, isBootstrapping: false }) }));
const toast = vi.hoisted(() => ({ showToast: vi.fn() }));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => toast }));
vi.mock("../components/ThemeToggle/ThemeToggle", () => ({ ThemeToggle: () => null }));

import { registerUser } from "../api/auth";
import { ApiError } from "../api/client";
import { RegisterPage } from "./RegisterPage";

const VALID = { name: "Jane Doe", email: "jane@example.com", password: "Secret123", confirmPassword: "Secret123" };

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/register"]}>
      <Routes>
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/login" element={<p>Login page</p>} />
      </Routes>
    </MemoryRouter>
  );
}

function fillForm(values = {}) {
  const { name, email, password, confirmPassword } = { ...VALID, ...values };
  fireEvent.change(screen.getByLabelText("Full name"), { target: { value: name } });
  fireEvent.change(screen.getByLabelText("Email address"), { target: { value: email } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: password } });
  fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: confirmPassword } });
}

const submit = () => fireEvent.click(screen.getByRole("button", { name: "Create account" }));

describe("RegisterPage", () => {
  beforeEach(() => {
    registerUser.mockReset();
    toast.showToast.mockReset();
  });

  it("renders name, email, password and confirm fields and no role field", () => {
    const { container } = renderPage();
    expect(screen.getByLabelText("Full name")).toBeInTheDocument();
    expect(screen.getByLabelText("Email address")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByLabelText("Confirm password")).toBeInTheDocument();

    expect(screen.queryByLabelText(/role/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(container.querySelector('[name="role"]')).toBeNull();
    expect(container.querySelectorAll("input")).toHaveLength(4);
  });

  it("explains that new accounts are Factory Supervisors and links back to sign in", () => {
    renderPage();
    expect(
      screen.getByText("New accounts are created as Factory Supervisor. A Quality Engineer can change your role.")
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to sign in" })).toHaveAttribute("href", "/login");
  });

  it.each([
    ["mismatched passwords", { confirmPassword: "Different123" }, "Passwords do not match."],
    ["a password shorter than 8 characters", { password: "Short7!", confirmPassword: "Short7!" }, "Password must be between 8 and 72 characters."],
    ["a password longer than 72 characters", { password: "x".repeat(73), confirmPassword: "x".repeat(73) }, "Password must be between 8 and 72 characters."],
    ["an invalid email", { email: "not-an-email" }, "Enter a valid email address."],
    ["an empty name", { name: "   " }, "Full name is required."],
  ])("rejects %s without calling the API", (_, values, message) => {
    renderPage();
    fillForm(values);
    submit();
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(registerUser).not.toHaveBeenCalled();
  });

  it("accepts a password of exactly 72 characters", async () => {
    registerUser.mockResolvedValue({});
    renderPage();
    fillForm({ password: "x".repeat(72), confirmPassword: "x".repeat(72) });
    submit();
    await screen.findByText("Login page");
    expect(registerUser).toHaveBeenCalledTimes(1);
  });

  it("submits exactly name, email and password (never a role) and redirects to /login", async () => {
    registerUser.mockResolvedValue({ id: 9, role: "factory_supervisor" });
    renderPage();
    fillForm();
    submit();

    await screen.findByText("Login page");
    expect(registerUser).toHaveBeenCalledTimes(1);
    const payload = registerUser.mock.calls[0][0];
    expect(payload).toEqual({ name: "Jane Doe", email: "jane@example.com", password: "Secret123" });
    expect(Object.keys(payload)).not.toContain("role");
    expect(toast.showToast).toHaveBeenCalledWith(expect.objectContaining({ type: "success" }));
  });

  it("shows the duplicate-email message on 409 and disables submit while submitting", async () => {
    let rejectRequest;
    registerUser.mockReturnValue(new Promise((_, reject) => (rejectRequest = reject)));
    renderPage();
    fillForm();
    submit();

    const button = screen.getByRole("button", { name: "Create account" });
    expect(button).toBeDisabled();

    rejectRequest(new ApiError(409, "Email already registered"));
    expect(await screen.findByText("An account with this email already exists.")).toBeInTheDocument();
    expect(screen.queryByText("Email already registered")).not.toBeInTheDocument();
    await waitFor(() => expect(button).not.toBeDisabled());
    expect(screen.queryByText("Login page")).not.toBeInTheDocument();
  });

  it("shows a generic message for server errors instead of raw details", async () => {
    registerUser.mockRejectedValue(new ApiError(500, "psycopg.OperationalError: connection refused"));
    renderPage();
    fillForm();
    submit();
    expect(await screen.findByText("Something went wrong. Please try again.")).toBeInTheDocument();
    expect(screen.queryByText(/psycopg/)).not.toBeInTheDocument();
  });

  it("shows the normalized message for network errors", async () => {
    registerUser.mockRejectedValue(new ApiError(0, "Unable to reach the server. Check your connection and try again."));
    renderPage();
    fillForm();
    submit();
    expect(await screen.findByText("Unable to reach the server. Check your connection and try again.")).toBeInTheDocument();
  });
});
