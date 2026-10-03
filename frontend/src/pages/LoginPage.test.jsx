import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../auth/useAuth", () => ({
  useAuth: () => ({ login: vi.fn(), isAuthenticated: false, isBootstrapping: false }),
}));
vi.mock("../components/ThemeToggle/ThemeToggle", () => ({ ThemeToggle: () => null }));

import { LoginPage } from "./LoginPage";

describe("LoginPage", () => {
  it("links to the registration page", () => {
    render(
      <MemoryRouter initialEntries={["/login"]}>
        <LoginPage />
      </MemoryRouter>
    );
    expect(screen.getByRole("link", { name: "Create an account" })).toHaveAttribute("href", "/register");
  });
});
