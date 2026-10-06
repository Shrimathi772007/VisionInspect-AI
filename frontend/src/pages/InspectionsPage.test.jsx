import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../auth/useAuth", () => ({ useAuth: () => ({ user: { name: "Ada", role: "quality_engineer" } }) }));
vi.mock("../hooks/useInspections", () => ({ useInspections: vi.fn() }));
vi.mock("../hooks/useProducts", () => ({
  useProducts: () => ({ getProductById: () => ({ id: 1, product_name: "Tile A", product_code: "TILE-1" }) }),
}));
vi.mock("../api/inspections", () => ({ deleteInspection: vi.fn() }));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));

import { useInspections } from "../hooks/useInspections";
import { InspectionsPage } from "./InspectionsPage";

const NOW = "2026-10-04T10:00:00Z";
const row = (overrides) => ({ product_id: 1, status: "pending", source: "upload", created_at: NOW, ...overrides });

function renderList(inspections) {
  useInspections.mockReturnValue({ inspections, isLoading: false, error: null, refetch: vi.fn() });
  return render(
    <MemoryRouter>
      <InspectionsPage />
    </MemoryRouter>
  );
}

const rowFor = (id) => within(screen.getByRole("link", { name: new RegExp(`#${id}\\b`) }));

describe("InspectionsPage manual review indicator", () => {
  beforeEach(() => {
    useInspections.mockReset();
  });

  it("marks an inspection that needs manual review, and only that one", () => {
    renderList([
      row({ id: 1, review_required: true, quality_decision: "MANUAL_REVIEW" }),
      row({ id: 2, review_required: false, quality_decision: "PASS" }),
      row({ id: 3 }), // an older row without the new fields
    ]);

    expect(rowFor(1).getByText("MANUAL REVIEW")).toBeInTheDocument();
    expect(rowFor(2).queryByText("MANUAL REVIEW")).not.toBeInTheDocument();
    expect(rowFor(3).queryByText("MANUAL REVIEW")).not.toBeInTheDocument();
  });

  it("renders unknown quality decisions and review values without crashing", () => {
    renderList([row({ id: 4, review_required: null, quality_decision: "SOMETHING_NEW" })]);
    expect(rowFor(4).getByText("User Upload")).toBeInTheDocument();
  });
});

describe("InspectionsPage severity badge", () => {
  beforeEach(() => {
    useInspections.mockReset();
  });

  it("shows the severity level when there is one, and nothing otherwise", () => {
    renderList([
      row({ id: 5, severity_level: "Critical" }),
      row({ id: 6, severity_level: null }),
      row({ id: 7, severity_level: "Unheard-of" }),
    ]);

    expect(rowFor(5).getByText("Severity: Critical")).toBeInTheDocument();
    expect(rowFor(6).queryByText(/Severity:/)).not.toBeInTheDocument();
    expect(rowFor(7).getByText("Severity: Unheard-of")).toBeInTheDocument();
  });

  it("gives each row's chevron its fixed-size class and a 16px icon", () => {
    renderList([row({ id: 8, review_required: true, quality_decision: "MANUAL_REVIEW" }), row({ id: 9 })]);

    for (const id of [8, 9]) {
      const chevron = screen.getByRole("link", { name: new RegExp(`#${id}\\b`) }).querySelector("svg.lucide-chevron-right");
      expect(chevron.getAttribute("class")).toMatch(/chevron/);
      expect(chevron).toHaveAttribute("width", "16");
      expect(chevron).toHaveAttribute("height", "16");
    }
  });
});
