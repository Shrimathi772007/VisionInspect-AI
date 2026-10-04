import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// The real routing table, ProtectedRoute, RoleRoute and ToastProvider (as in App.test.jsx); only the auth
// state, the layout chrome and the page bodies are stubbed.
const auth = vi.hoisted(() => ({ user: null, isAuthenticated: false, isBootstrapping: false }));
vi.mock("./auth/useAuth", () => ({ useAuth: () => auth }));
vi.mock("./layouts/AppLayout", async () => {
  const { Outlet } = await import("react-router-dom");
  return { AppLayout: () => <Outlet /> };
});
vi.mock("./pages/DashboardPage", () => ({ DashboardPage: () => <p>Dashboard page</p> }));
vi.mock("./pages/CameraSimulationPage", () => ({ CameraSimulationPage: () => <p>Camera page</p> }));
vi.mock("./pages/LoginPage", () => ({ LoginPage: () => <p>Login page</p> }));

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

describe("/camera route", () => {
  beforeEach(() => {
    auth.user = null;
    auth.isAuthenticated = false;
    auth.isBootstrapping = false;
  });

  it("redirects a factory supervisor to the dashboard", async () => {
    auth.user = { id: 2, name: "Bob", role: "factory_supervisor" };
    auth.isAuthenticated = true;
    renderAt("/camera");
    expect(await screen.findByText("Dashboard page")).toBeInTheDocument();
    expect(screen.queryByText("Camera page")).not.toBeInTheDocument();
  });

  it("shows the camera page to a quality engineer", () => {
    auth.user = { id: 1, name: "Ada", role: "quality_engineer" };
    auth.isAuthenticated = true;
    renderAt("/camera");
    expect(screen.getByText("Camera page")).toBeInTheDocument();
  });
});
