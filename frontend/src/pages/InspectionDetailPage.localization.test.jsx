import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../api/inspections", () => ({
  getInspection: vi.fn(),
  getInspectionReport: vi.fn(),
  getInspectionImageObjectUrl: vi.fn(),
  getInspectionHeatmapBlob: vi.fn(),
  deleteInspection: vi.fn(),
}));
vi.mock("../auth/useAuth", () => ({ useAuth: () => ({ user: { name: "Ada", role: "quality_engineer" } }) }));
vi.mock("../hooks/useProducts", () => ({
  useProducts: () => ({
    isLoading: false,
    getProductById: () => ({ id: 1, product_name: "Tile A", product_code: "TILE-1" }),
  }),
}));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
// The real viewer's zoom/pan is irrelevant here; the overlay it is given is rendered as-is.
vi.mock("../components/ImageViewer/ImageViewer", () => ({
  ImageViewer: ({ overlay }) => <div data-testid="image-viewer">{overlay}</div>,
}));

import {
  getInspection,
  getInspectionHeatmapBlob,
  getInspectionImageObjectUrl,
  getInspectionReport,
} from "../api/inspections";
import { InspectionDetailPage } from "./InspectionDetailPage";

const NOW = "2026-10-04T10:00:00Z";
const HEURISTIC_NOTE =
  "Confidence is a margin-based heuristic from the distance between the anomaly score and the decision threshold. It is not a calibrated probability.";
const LOCALIZATION_NOTE =
  "Anomaly-based: regions are derived from the model's anomaly heatmap. Not an object-detection model.";

const DEFECTIVE_UPLOAD = {
  id: 21,
  product_id: 1,
  status: "pending",
  source: "upload",
  product_category: "tile",
  inspection_date: NOW,
  created_at: NOW,
  defect_category: null,
  ai_prediction: "defective",
  ai_reconstruction_error: 2.4,
  ai_threshold: 1.739,
  ai_model_name: "knn_l23_256",
  ai_inference_time_ms: 160,
  processing_time_ms: 400,
  severity_score: null,
  severity_level: null,
  quality_risk: null,
  quality_decision: "FAIL",
  quality_assessment: "Defective product requires review.",
  quality_recommendation: "Review the detected anomaly before approving the product.",
  ai_confidence: 0.9612,
  ai_reliability: "high",
  review_required: false,
  review_reason: null,
  model_gate: "EXCELLENT",
  has_heatmap: true,
  localization: {
    method: "anomaly_map_threshold_v1",
    boxes: [
      { x: 0.25, y: 0.1, w: 0.2, h: 0.15, peak: 3.21, area_fraction: 0.02 },
      { x: 0.6, y: 0.55, w: 0.1, h: 0.05, peak: 2.05, area_fraction: 0.004 },
    ],
    area_pct: 3.2,
    centroid: [0.35, 0.175],
    mask_rule: "threshold",
    analysed_region: [0, 0, 1, 1],
  },
};

const GOOD_CROP_UPLOAD = {
  ...DEFECTIVE_UPLOAD,
  id: 22,
  ai_prediction: "good",
  ai_reconstruction_error: 1.1,
  quality_decision: "PASS",
  localization: {
    method: "anomaly_map_threshold_v1",
    boxes: [],
    area_pct: 0,
    centroid: null,
    mask_rule: null,
    analysed_region: [0.0625, 0.0625, 0.9375, 0.9375],
  },
};

function renderPage(id) {
  return render(
    <MemoryRouter initialEntries={[`/inspections/${id}`]}>
      <Routes>
        <Route path="/inspections/:id" element={<InspectionDetailPage />} />
      </Routes>
    </MemoryRouter>
  );
}

const aiCard = () => within(screen.getByRole("heading", { name: "AI Prediction" }).parentElement);

async function loaded() {
  await screen.findByRole("heading", { name: /Defect localization/ });
}

describe("InspectionDetailPage localization, confidence and manual review", () => {
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

  beforeEach(() => {
    URL.createObjectURL.mockClear();
    URL.revokeObjectURL.mockClear();
    getInspection.mockReset().mockResolvedValue(DEFECTIVE_UPLOAD);
    getInspectionReport.mockReset().mockResolvedValue({
      report_summary: { overall_result: "FAIL", report_status: "COMPLETE", summary: "Defective." },
    });
    getInspectionImageObjectUrl.mockReset().mockResolvedValue("blob:image");
    getInspectionHeatmapBlob.mockReset().mockResolvedValue(new Blob(["png"], { type: "image/png" }));
  });

  describe("heatmap", () => {
    it("fetches the heatmap blob with the inspection id, shows it, and revokes the object URL on unmount", async () => {
      const { unmount } = renderPage(21);
      await loaded();

      const overlay = await screen.findByTestId("heatmap-overlay");
      expect(getInspectionHeatmapBlob).toHaveBeenCalledWith("21");
      expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
      expect(overlay).toHaveAttribute("src", "blob:heatmap");
      expect(overlay.style.opacity).toBe("0.65");

      unmount();
      expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:heatmap");
    });

    it("a failed heatmap shows a message and the rest of the page still renders", async () => {
      getInspectionHeatmapBlob.mockRejectedValue(new Error("boom"));
      renderPage(21);
      await loaded();

      expect(await screen.findByText("Heatmap couldn't be loaded.")).toBeInTheDocument();
      expect(screen.queryByTestId("heatmap-overlay")).not.toBeInTheDocument();
      expect(screen.getAllByTestId("defect-region")).toHaveLength(2);
      expect(aiCard().getByText("96.1%")).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Quality Assessment" })).toBeInTheDocument();
    });

    it("the opacity control changes the overlay opacity", async () => {
      renderPage(21);
      await loaded();
      const overlay = await screen.findByTestId("heatmap-overlay");

      fireEvent.click(screen.getByRole("button", { name: "High" }));
      expect(overlay.style.opacity).toBe("1");
      expect(screen.getByRole("button", { name: "High" })).toHaveAttribute("aria-pressed", "true");
    });

    it("has_heatmap false hides the toggles, says so, and never fetches", async () => {
      getInspection.mockResolvedValue({ ...DEFECTIVE_UPLOAD, has_heatmap: false });
      renderPage(21);
      await loaded();

      expect(screen.getByText("Heatmap not available for this inspection.")).toBeInTheDocument();
      expect(screen.queryByRole("checkbox", { name: "Heatmap" })).not.toBeInTheDocument();
      expect(screen.queryByRole("checkbox", { name: "Defect regions" })).not.toBeInTheDocument();
      expect(getInspectionHeatmapBlob).not.toHaveBeenCalled();
    });
  });

  describe("defect regions", () => {
    it("draws each box at its normalised position as percentages, labelled by rank and peak", async () => {
      renderPage(21);
      await loaded();

      const [first, second] = screen.getAllByTestId("defect-region");
      expect(first.style.left).toBe("25%");
      expect(first.style.top).toBe("10%");
      expect(first.style.width).toBe("20%");
      expect(first.style.height).toBe("15%");
      expect(first).toHaveTextContent("1 · 3.21");
      expect(second.style.left).toBe("60%");
      expect(second.style.top).toBe("55%");
      expect(second.style.width).toBe("10%");
      expect(second.style.height).toBe("5%");
      expect(second).toHaveTextContent("2 · 2.05");
    });

    it("toggles hide and show the overlays", async () => {
      renderPage(21);
      await loaded();
      await screen.findByTestId("heatmap-overlay");

      fireEvent.click(screen.getByRole("checkbox", { name: "Defect regions" }));
      expect(screen.queryAllByTestId("defect-region")).toHaveLength(0);
      fireEvent.click(screen.getByRole("checkbox", { name: "Heatmap" }));
      expect(screen.queryByTestId("heatmap-overlay")).not.toBeInTheDocument();

      fireEvent.click(screen.getByRole("checkbox", { name: "Defect regions" }));
      fireEvent.click(screen.getByRole("checkbox", { name: "Heatmap" }));
      expect(screen.getAllByTestId("defect-region")).toHaveLength(2);
      expect(screen.getByTestId("heatmap-overlay")).toBeInTheDocument();
    });

    it("explains the regions in words with the anomaly-based note", async () => {
      renderPage(21);
      await loaded();

      expect(screen.getByText(LOCALIZATION_NOTE)).toBeInTheDocument();
      expect(screen.getByText("Affected area 3.2% of the image")).toBeInTheDocument();
      expect(screen.getByText(/Centred in the upper centre of the image \(35% from the left, 18% from the top\)/)).toBeInTheDocument();
    });

    it("shows the analysed-region note and shading only when the model did not see the whole image", async () => {
      renderPage(21);
      await loaded();
      expect(screen.queryByTestId("analysed-region-note")).not.toBeInTheDocument();
      expect(screen.queryByTestId("analysed-region")).not.toBeInTheDocument();
    });

    it("a good prediction says no region is above the threshold, keeps the heatmap toggle, and shades the crop", async () => {
      getInspection.mockResolvedValue(GOOD_CROP_UPLOAD);
      renderPage(22);
      await loaded();

      expect(screen.getByText("No anomalous region above the threshold.")).toBeInTheDocument();
      expect(screen.getByRole("checkbox", { name: "Heatmap" })).toBeChecked();
      expect(screen.queryAllByTestId("defect-region")).toHaveLength(0);
      expect(screen.getByTestId("analysed-region-note")).toHaveTextContent("Edges outside the analysed region");
      const region = screen.getByTestId("analysed-region");
      expect(region.style.left).toBe("6.25%");
      expect(region.style.top).toBe("6.25%");
      expect(region.style.width).toBe("87.5%");
      expect(region.style.height).toBe("87.5%");
    });
  });

  describe("AI card", () => {
    it.each([
      ["high", "High reliability"],
      ["medium", "Medium reliability"],
      ["low", "Low reliability"],
    ])("shows the confidence percentage with the %s reliability badge and the heuristic note", async (reliability, label) => {
      getInspection.mockResolvedValue({ ...DEFECTIVE_UPLOAD, ai_reliability: reliability });
      renderPage(21);
      await loaded();

      expect(aiCard().getByText("96.1%")).toBeInTheDocument();
      expect(aiCard().getByText(label)).toBeInTheDocument();
      expect(aiCard().getByText(HEURISTIC_NOTE)).toBeInTheDocument();
      // Existing rows are kept.
      expect(aiCard().getByText("Anomaly score")).toBeInTheDocument();
      expect(aiCard().getByText("Threshold")).toBeInTheDocument();
    });

    it.each([
      ["EXCELLENT", "Production ready (Excellent)"],
      ["GOOD", "Good"],
      ["ACCEPTABLE", "Acceptable"],
      ["NOT_PRODUCTION_READY", "Not production ready"],
      [null, "Not available"],
      ["SOMETHING_NEW", "SOMETHING_NEW"],
    ])("shows the model gate %s as %s", async (gate, label) => {
      getInspection.mockResolvedValue({ ...DEFECTIVE_UPLOAD, model_gate: gate });
      renderPage(21);
      await loaded();

      const gateRow = aiCard().getByText("Model gate").parentElement;
      expect(within(gateRow).getByText(label)).toBeInTheDocument();
    });

    it("shows Not available when there is no confidence", async () => {
      getInspection.mockResolvedValue({ ...DEFECTIVE_UPLOAD, ai_confidence: null, ai_reliability: null });
      renderPage(21);
      await loaded();

      const row = aiCard().getByText("Confidence").parentElement;
      expect(within(row).getByText("Not available")).toBeInTheDocument();
    });
  });

  describe("manual review", () => {
    it("shows the banner with the reasons in plain words and the MANUAL REVIEW decision badge", async () => {
      getInspection.mockResolvedValue({
        ...DEFECTIVE_UPLOAD,
        review_required: true,
        review_reason: "low confidence; category model not production ready",
        quality_decision: "MANUAL_REVIEW",
      });
      renderPage(21);
      await loaded();

      const banner = screen.getByTestId("manual-review-banner");
      expect(banner).toHaveTextContent("Manual review required");
      expect(banner).toHaveTextContent("Reason: Low confidence; Category model not production ready");
      expect(banner).toHaveTextContent("The AI result is shown for guidance only.");
      const qualityCard = within(screen.getByRole("heading", { name: "Quality Assessment" }).parentElement);
      expect(qualityCard.getByText("MANUAL REVIEW")).toBeInTheDocument();
    });

    it.each([false, null, undefined])("shows no banner when review_required is %s", async (value) => {
      getInspection.mockResolvedValue({ ...DEFECTIVE_UPLOAD, review_required: value });
      renderPage(21);
      await loaded();

      expect(screen.queryByTestId("manual-review-banner")).not.toBeInTheDocument();
      expect(screen.queryByText("Manual review required")).not.toBeInTheDocument();
    });

    it("an unknown quality decision renders as-is without crashing", async () => {
      getInspection.mockResolvedValue({ ...DEFECTIVE_UPLOAD, quality_decision: "ESCALATED" });
      renderPage(21);
      await loaded();

      const qualityCard = within(screen.getByRole("heading", { name: "Quality Assessment" }).parentElement);
      await waitFor(() => expect(qualityCard.getByText("ESCALATED")).toBeInTheDocument());
    });
  });
});
