import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// The real routing table, ProtectedRoute, RoleRoute and ToastProvider; only the auth state,
// the layout chrome and the page bodies are stubbed.
const auth = vi.hoisted(() => ({ user: null, isAuthenticated: false, isBootstrapping: false }));
vi.mock("./auth/useAuth", () => ({ useAuth: () => auth }));
vi.mock("./layouts/AppLayout", async () => {
  const { Outlet } = await import("react-router-dom");
  return { AppLayout: () => <Outlet /> };
});
vi.mock("./pages/LoginPage", () => ({ LoginPage: () => <p>Login page</p> }));
vi.mock("./pages/RegisterPage", () => ({ RegisterPage: () => <p>Register page</p> }));
vi.mock("./pages/DashboardPage", () => ({ DashboardPage: () => <p>Dashboard page</p> }));
vi.mock("./pages/ProductsPage", () => ({ ProductsPage: () => <p>Products page</p> }));
vi.mock("./pages/InspectionsPage", () => ({ InspectionsPage: () => <p>Inspections page</p> }));
vi.mock("./pages/InspectionUploadPage", () => ({ InspectionUploadPage: () => <p>Upload page</p> }));
vi.mock("./pages/InspectionDetailPage", () => ({ InspectionDetailPage: () => <p>Inspection detail page</p> }));
vi.mock("./pages/DatasetBrowserPage", () => ({ DatasetBrowserPage: () => <p>Dataset page</p> }));
vi.mock("./pages/UsersPage", () => ({ UsersPage: () => <p>Users page</p> }));

import App from "./App";
import { ToastProvider } from "./components/Toast/ToastProvider";

function renderAt(path) {
  return render(
    <ToastProvider>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </ToastProvider>
  );
}

function signInAs(role) {
  auth.user = { id: 1, name: "Test User", role };
  auth.isAuthenticated = true;
}

describe("App route guards", () => {
  beforeEach(() => {
    auth.user = null;
    auth.isAuthenticated = false;
    auth.isBootstrapping = false;
  });

  it("redirects a factory supervisor from /inspections/upload to /dashboard with the toast", async () => {
    signInAs("factory_supervisor");
    renderAt("/inspections/upload");

    expect(await screen.findByText("Dashboard page")).toBeInTheDocument();
    expect(screen.queryByText("Upload page")).not.toBeInTheDocument();
    expect(screen.getAllByText("Quality Engineer access required")).toHaveLength(1);
  });

  it("shows the upload page to a quality engineer", () => {
    signInAs("quality_engineer");
    renderAt("/inspections/upload");

    expect(screen.getByText("Upload page")).toBeInTheDocument();
    expect(screen.queryByText("Quality Engineer access required")).not.toBeInTheDocument();
  });

  it.each([
    ["/inspections", "Inspections page"],
    ["/inspections/42", "Inspection detail page"],
  ])("keeps %s reachable for a factory supervisor", (path, page) => {
    signInAs("factory_supervisor");
    renderAt(path);

    expect(screen.getByText(page)).toBeInTheDocument();
    expect(screen.queryByText("Quality Engineer access required")).not.toBeInTheDocument();
  });

  it("still sends a signed-out visitor of /inspections/upload to /login", () => {
    renderAt("/inspections/upload");
    expect(screen.getByText("Login page")).toBeInTheDocument();
  });
});
