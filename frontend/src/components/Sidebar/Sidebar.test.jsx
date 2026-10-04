import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../ThemeToggle/ThemeToggle", () => ({ ThemeToggle: () => null }));

import { Sidebar } from "./Sidebar";

function renderSidebar(role) {
  return render(
    <MemoryRouter>
      <Sidebar
        collapsed={false}
        onToggleCollapsed={() => {}}
        mobileOpen={false}
        onCloseMobile={() => {}}
        user={{ id: 1, name: "Test User", role }}
        onLogout={() => {}}
      />
    </MemoryRouter>
  );
}

describe("Sidebar", () => {
  it("shows the Users item to a quality engineer", () => {
    renderSidebar("quality_engineer");
    expect(screen.getByRole("link", { name: "Users" })).toHaveAttribute("href", "/users");
    expect(screen.getByRole("link", { name: "Upload Inspection" })).toBeInTheDocument();
  });

  it("hides the Users item from a factory supervisor", () => {
    renderSidebar("factory_supervisor");
    expect(screen.queryByRole("link", { name: "Users" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Upload Inspection" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Dashboard" })).toBeInTheDocument();
  });

  it.each(["quality_engineer", "factory_supervisor"])("shows the Model Performance item to a %s", (role) => {
    renderSidebar(role);
    expect(screen.getByRole("link", { name: "Model Performance" })).toHaveAttribute("href", "/models");
  });
});
