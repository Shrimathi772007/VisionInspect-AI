import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

// The localization is anomaly-based (regions derived from the anomaly heatmap). No page may call it
// YOLO or an object-detection model; the only allowed mention is the explicit negation.
vi.mock("../api/inspections", () => ({
  getInspection: vi.fn(),
  getInspectionReport: vi.fn(),
  getInspectionImageObjectUrl: vi.fn(),
  getInspectionHeatmapBlob: vi.fn(),
  deleteInspection: vi.fn(),
  uploadInspection: vi.fn(),
}));
vi.mock("../api/ai", () => ({ getAiModels: vi.fn() }));
vi.mock("../auth/useAuth", () => ({ useAuth: () => ({ user: { name: "Ada", role: "quality_engineer" } }) }));
vi.mock("../hooks/useProducts", () => ({
  useProducts: () => ({
    products: [],
    isLoading: false,
    getProductById: () => ({ id: 1, product_name: "Tile A", product_code: "TILE-1" }),
  }),
}));
vi.mock("../hooks/useInspections", () => ({ useInspections: vi.fn() }));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("../components/ImageViewer/ImageViewer", () => ({
  ImageViewer: ({ overlay }) => <div>{overlay}</div>,
}));

import { getInspection, getInspectionHeatmapBlob, getInspectionImageObjectUrl, getInspectionReport } from "../api/inspections";
import { getAiModels } from "../api/ai";
import { useInspections } from "../hooks/useInspections";
import { InspectionDetailPage } from "./InspectionDetailPage";
import { InspectionsPage } from "./InspectionsPage";
import { ModelPerformancePage } from "./ModelPerformancePage";

const INSPECTION = {
  id: 9,
  product_id: 1,
  status: "pending",
  source: "upload",
  product_category: "wood",
  inspection_date: "2026-10-04T10:00:00Z",
  created_at: "2026-10-04T10:00:00Z",
  ai_prediction: "defective",
  ai_reconstruction_error: 2.5,
  ai_threshold: 1.88,
  ai_model_name: "wrn50_patchcore_full256",
  quality_decision: "MANUAL_REVIEW",
  ai_confidence: 0.94,
  ai_reliability: "medium",
  review_required: true,
  review_reason: "category model not production ready",
  model_gate: "NOT_PRODUCTION_READY",
  has_heatmap: true,
  localization: {
    method: "anomaly_map_threshold_v1",
    boxes: [{ x: 0.1, y: 0.2, w: 0.3, h: 0.2, peak: 2.6, area_fraction: 0.05 }],
    area_pct: 5,
    centroid: [0.25, 0.3],
    mask_rule: "threshold",
    analysed_region: [0.0625, 0.0625, 0.9375, 0.9375],
  },
};

function assertTerminology() {
  const text = document.body.textContent;
  expect(text).not.toMatch(/yolo/i);
  const withoutNegation = text.replaceAll("Not an object-detection model", "");
  expect(withoutNegation).not.toMatch(/object[- ]detection/i);
}

describe("UI terminology", () => {
  const realCreate = URL.createObjectURL;
  const realRevoke = URL.revokeObjectURL;
  beforeAll(() => {
    URL.createObjectURL = vi.fn(() => "blob:heatmap");
    URL.revokeObjectURL = vi.fn();
  });
  afterAll(() => {
    URL.createObjectURL = realCreate;
    URL.revokeObjectURL = realRevoke;
  });

  it("the inspection detail page never says YOLO or calls the localization an object-detection model", async () => {
    getInspection.mockResolvedValue(INSPECTION);
    getInspectionReport.mockResolvedValue({ report_summary: { overall_result: "MANUAL_REVIEW", report_status: "COMPLETE", summary: "s" } });
    getInspectionImageObjectUrl.mockResolvedValue("blob:image");
    getInspectionHeatmapBlob.mockResolvedValue(new Blob(["png"]));
    render(
      <MemoryRouter initialEntries={["/inspections/9"]}>
        <Routes>
          <Route path="/inspections/:id" element={<InspectionDetailPage />} />
        </Routes>
      </MemoryRouter>
    );
    await screen.findByText("Anomaly-based: regions are derived from the model's anomaly heatmap. Not an object-detection model.");
    await screen.findByTestId("heatmap-overlay");
    assertTerminology();
  });

  it("the inspections list and the model performance page never say YOLO", async () => {
    useInspections.mockReturnValue({ inspections: [INSPECTION], isLoading: false, error: null, refetch: vi.fn() });
    getAiModels.mockResolvedValue([
      {
        category: "wood",
        model_name: "wrn50_patchcore_full256",
        family: "patchcore_wrn50",
        input_mode: "full256",
        input_size: [256, 256],
        threshold: 1.88,
        gate: "NOT_PRODUCTION_READY",
        final_test_recall: 0.98,
        final_test_fpr: 0.37,
        final_test_auroc: 0.99,
        final_test_average_precision: 0.99,
      },
    ]);
    render(
      <MemoryRouter>
        <InspectionsPage />
        <ModelPerformancePage />
      </MemoryRouter>
    );
    await screen.findByText("Wood");
    assertTerminology();
  });
});
