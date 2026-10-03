import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

const auth = vi.hoisted(() => ({ user: null }));
vi.mock("../../auth/useAuth", () => ({ useAuth: () => auth }));

import { ToastProvider } from "../Toast/ToastProvider";
import { RoleRoute } from "./RoleRoute";

function renderAt(path) {
  return render(
    <ToastProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route
            path="/users"
            element={
              <RoleRoute allow={["quality_engineer"]}>
                <p>Users page</p>
              </RoleRoute>
            }
          />
          <Route path="/dashboard" element={<p>Dashboard page</p>} />
        </Routes>
      </MemoryRouter>
    </ToastProvider>
  );
}

describe("RoleRoute", () => {
  beforeEach(() => {
    auth.user = null;
  });

  it("redirects a factory supervisor from /users to /dashboard with an info toast", async () => {
    auth.user = { id: 2, name: "Bob", role: "factory_supervisor" };
    renderAt("/users");

    expect(await screen.findByText("Dashboard page")).toBeInTheDocument();
    expect(screen.queryByText("Users page")).not.toBeInTheDocument();
    expect(screen.getAllByText("Quality Engineer access required")).toHaveLength(1);
  });

  it("renders the page for a quality engineer", () => {
    auth.user = { id: 1, name: "Ada", role: "quality_engineer" };
    renderAt("/users");

    expect(screen.getByText("Users page")).toBeInTheDocument();
    expect(screen.queryByText("Quality Engineer access required")).not.toBeInTheDocument();
  });
});
