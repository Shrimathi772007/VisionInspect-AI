import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ApiError } from "../api/client";

const { mockLogin } = vi.hoisted(() => ({ mockLogin: vi.fn() }));

vi.mock("../auth/useAuth", () => ({
  useAuth: () => ({ login: mockLogin, isAuthenticated: false, isBootstrapping: false }),
}));
vi.mock("../components/ThemeToggle/ThemeToggle", () => ({ ThemeToggle: () => null }));

import { LoginPage } from "./LoginPage";

const TOO_MANY = "Too many failed login attempts. Please wait and try again later.";

function renderPage() {
  render(
    <MemoryRouter initialEntries={["/login"]}>
      <LoginPage />
    </MemoryRouter>
  );
}

function submit() {
  fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "someone@example.com" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "WrongPass123" } });
  fireEvent.submit(screen.getByLabelText("Email address").closest("form"));
}

describe("LoginPage", () => {
  beforeEach(() => {
    mockLogin.mockReset();
  });

  it("links to the registration page", () => {
    renderPage();
    expect(screen.getByRole("link", { name: "Create an account" })).toHaveAttribute("href", "/register");
  });

  it("shows the rate-limit message for a 429", async () => {
    mockLogin.mockRejectedValue(new ApiError(429, TOO_MANY));
    renderPage();
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent(TOO_MANY);
  });

  it("shows the clean rate-limit message when the 429 had no readable detail", async () => {
    mockLogin.mockRejectedValue(new ApiError(429, "Request failed with status 429"));
    renderPage();
    submit();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(TOO_MANY);
    expect(alert).not.toHaveTextContent("429");
  });

  it("still shows the server message for wrong credentials", async () => {
    mockLogin.mockRejectedValue(new ApiError(401, "Incorrect email or password"));
    renderPage();
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect email or password");
  });
});
