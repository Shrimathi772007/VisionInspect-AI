import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../api/ai", () => ({ getAiModels: vi.fn() }));

import { getAiModels } from "../api/ai";
import { LOCALIZATION_FOOTNOTE, ModelPerformancePage } from "./ModelPerformancePage";
import { MVTEC_CATEGORIES, categoryLabel } from "../constants/mvtecCategories";

const GATES = ["EXCELLENT", "GOOD", "ACCEPTABLE", "NOT_PRODUCTION_READY"];

// 15 rows shaped like GET /ai/models; the ResNet-18 rows carry the post-hoc AP addendum value.
const MODELS = MVTEC_CATEGORIES.map((category, index) => {
  const wrn = index % 2 === 0;
  return {
    category,
    model_name: wrn ? "wrn50_patchcore_crop224" : "knn_l23_256",
    family: wrn ? "patchcore_wrn50" : "patch_anomaly",
    input_mode: wrn ? "crop224" : "resize",
    input_size: wrn ? [224, 224] : [256, 256],
    threshold: 1.5,
    gate: GATES[index % 4],
    final_test_recall: 0.9166666,
    final_test_fpr: 0.0454545,
    final_test_auroc: 0.99711,
    final_test_average_precision: wrn ? 0.99515 : 0.98539,
  };
});

async function tableRows(name) {
  const table = await screen.findByRole("table", { name });
  return within(table).getAllByRole("row").slice(1);
}

const modelRows = () => tableRows("Served models");
const localizationRows = () => tableRows("Localization evaluation");

function renderPage() {
  return render(
    <MemoryRouter>
      <ModelPerformancePage />
    </MemoryRouter>
  );
}

describe("ModelPerformancePage", () => {
  beforeEach(() => {
    getAiModels.mockReset();
  });

  it("renders one row per category from the API with its gate and metrics", async () => {
    getAiModels.mockResolvedValue(MODELS);
    renderPage();

    const rows = await modelRows();
    expect(rows).toHaveLength(15);
    const first = within(rows[0]);
    expect(first.getByText(categoryLabel(MODELS[0].category))).toBeInTheDocument();
    expect(first.getByText("WRN-50 PatchCore")).toBeInTheDocument();
    expect(first.getByText("centre crop 224")).toBeInTheDocument();
    expect(first.getByText("Production ready (Excellent)")).toBeInTheDocument();
    expect(first.getByText("91.7%")).toBeInTheDocument();
    expect(first.getByText("4.5%")).toBeInTheDocument();
    expect(first.getByText("0.997")).toBeInTheDocument();
    expect(first.getByText("0.995")).toBeInTheDocument();

    const second = within(rows[1]);
    expect(second.getByText("ResNet-18 patch model")).toBeInTheDocument();
    expect(second.getByText("whole image 256")).toBeInTheDocument();
    expect(second.getByText("Good")).toBeInTheDocument();
    expect(second.getByText("0.985")).toBeInTheDocument(); // ResNet-18 AP from the addendum
    expect(second.queryByText("—")).not.toBeInTheDocument();
    expect(within(rows[3]).getByText("Not production ready")).toBeInTheDocument();
  });

  it("shows the footnote and never a path or hash", async () => {
    getAiModels.mockResolvedValue(MODELS);
    renderPage();
    await modelRows();

    expect(screen.getByText(/Metrics are from each category's single final test on the MVTec AD test set\./)).toBeInTheDocument();
    expect(screen.getByText(/Excellent: recall >= 0\.90, F1 >= 0\.85, FPR <= 0\.10/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/ai_models|\.pt\b|sha256/i);
  });

  it("shows a ResNet-18 AP of 1.0 as 1.000 and a missing AP as a dash", async () => {
    getAiModels.mockResolvedValue([
      { ...MODELS[1], category: "leather", final_test_average_precision: 1.0 },
      { ...MODELS[3], category: "tile", final_test_average_precision: null },
      { ...MODELS[5], category: "cable", final_test_average_precision: undefined },
    ]);
    renderPage();

    const rows = await modelRows();
    expect(within(rows[0]).getByText("1.000")).toBeInTheDocument();
    expect(within(rows[1]).getByText("—")).toBeInTheDocument();
    expect(within(rows[2]).getByText("—")).toBeInTheDocument();
  });

  it("explains how AP was computed, including the post-hoc ResNet-18 values", async () => {
    getAiModels.mockResolvedValue(MODELS);
    renderPage();
    await modelRows();

    expect(
      screen.getByText(
        "AP is image-level (defective = positive). For the 6 ResNet-18 categories it was computed after the final " +
          "test from the saved scores; no model or threshold changed."
      )
    ).toBeInTheDocument();
  });

  it("renders an unknown or null gate safely", async () => {
    getAiModels.mockResolvedValue([{ ...MODELS[0], gate: null }, { ...MODELS[1], gate: "NEW_GATE" }]);
    renderPage();

    expect(await screen.findByText("Not available")).toBeInTheDocument();
    expect(screen.getByText("NEW_GATE")).toBeInTheDocument();
  });

  it("shows loading, then an error with Retry", async () => {
    getAiModels.mockRejectedValue(new Error("Server down"));
    renderPage();

    expect(screen.getByRole("status", { name: "Loading models" })).toBeInTheDocument();
    expect(await screen.findByText("Failed to load models. Server down")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("shows an empty state when no models are registered", async () => {
    getAiModels.mockResolvedValue([]);
    renderPage();

    expect(await screen.findByText("No models registered")).toBeInTheDocument();
  });

  it("shows box AP@0.5 (one-box-per-image AP in brackets) and pixel AUROC in a second table", async () => {
    getAiModels.mockResolvedValue([
      { ...MODELS[0], box_ap50: 0.12345, box_ap50_merged: 0.25, pixel_auroc: 0.9712 },
      { ...MODELS[1], box_ap50: 0, box_ap50_merged: 0, pixel_auroc: 1 },
    ]);
    renderPage();

    const rows = await localizationRows();
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText(categoryLabel(MODELS[0].category))).toBeInTheDocument();
    expect(rows[0]).toHaveTextContent("0.123 (0.250)");
    expect(within(rows[0]).getByText("0.971")).toBeInTheDocument();
    expect(rows[1]).toHaveTextContent("0.000 (0.000)");
    expect(within(rows[1]).getByText("1.000")).toBeInTheDocument();
  });

  it("shows a dash when the localization addendum or a category is missing", async () => {
    getAiModels.mockResolvedValue([
      { ...MODELS[0], box_ap50: null, box_ap50_merged: null, pixel_auroc: null },
      { ...MODELS[1] }, // an older API without the fields
    ]);
    renderPage();

    for (const row of await localizationRows()) {
      expect(within(row).getAllByText("—")).toHaveLength(2);
      expect(row).not.toHaveTextContent("(");
    }
  });

  it("explains the localization evaluation in a footnote", async () => {
    getAiModels.mockResolvedValue(MODELS);
    renderPage();
    await localizationRows();

    expect(screen.getByText(LOCALIZATION_FOOTNOTE)).toBeInTheDocument();
    expect(LOCALIZATION_FOOTNOTE).toMatch(/^Second scoring of final test sets for localization only; no model or threshold changed; anomaly-map boxes, not a trained detector\./);
    expect(LOCALIZATION_FOOTNOTE).toMatch(/excludes pixels outside the analysed area for the centre-crop 224 categories/);
  });
});
