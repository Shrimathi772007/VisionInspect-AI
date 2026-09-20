import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../api/inspections", () => ({
  getInspection: vi.fn(),
  getInspectionReport: vi.fn(),
  getInspectionImageObjectUrl: vi.fn(),
  deleteInspection: vi.fn(),
}));
vi.mock("../auth/useAuth", () => ({ useAuth: () => ({ user: { name: "Ada", role: "quality_engineer" } }) }));
vi.mock("../hooks/useProducts", () => ({
  useProducts: () => ({
    isLoading: false,
    getProductById: () => ({ id: 1, product_name: "Bottle A", product_code: "BOT-1" }),
  }),
}));
vi.mock("../components/Toast/ToastProvider", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("../components/ImageViewer/ImageViewer", () => ({ ImageViewer: () => <div data-testid="image-viewer" /> }));

import { getInspection, getInspectionImageObjectUrl, getInspectionReport } from "../api/inspections";
import { InspectionDetailPage } from "./InspectionDetailPage";

const NOW = "2026-09-20T10:00:00Z";

const AI_INSPECTION = {
  id: 5,
  product_id: 1,
  status: "defective",
  source: "mvtec_ad",
  dataset_category: "bottle",
  dataset_split: "test",
  dataset_defect_type: "broken_large",
  dataset_filename: "000.png",
  inspection_date: NOW,
  created_at: NOW,
  defect_category: "broken_large",
  ai_prediction: "defective",
  ai_reconstruction_error: 0.0041,
  ai_threshold: 0.0032,
  ai_model_name: "autoencoder",
  severity_score: null,
  severity_level: null,
  quality_risk: null,
  quality_decision: "FAIL",
  quality_assessment: "Ground truth and AI agree the part is defective.",
  quality_recommendation: "Quarantine the part.",
  processing_time_ms: 845.4,
  ai_inference_time_ms: 12300,
};

const UPLOAD_INSPECTION = {
  ...AI_INSPECTION,
  id: 6,
  source: "upload",
  dataset_category: null,
  dataset_split: null,
  dataset_defect_type: null,
  dataset_filename: null,
  defect_category: null,
  ai_prediction: null,
  ai_reconstruction_error: null,
  ai_threshold: null,
  ai_model_name: null,
  quality_decision: "NOT_ASSESSED",
  processing_time_ms: 210,
  ai_inference_time_ms: null,
};

const report = (overrides = {}) => ({
  report_summary: {
    overall_result: "FAIL",
    report_status: "COMPLETE",
    summary: "Ground truth and AI agree the part is defective.",
    ...overrides,
  },
});

function renderPage(id = 5) {
  return render(
    <MemoryRouter initialEntries={[`/inspections/${id}`]}>
      <Routes>
        <Route path="/inspections/:id" element={<InspectionDetailPage />} />
      </Routes>
    </MemoryRouter>
  );
}

// The report card: the heading's parent is the Card that holds its contents.
const reportCard = () => within(screen.getByRole("heading", { name: "Production Quality Report" }).parentElement);

async function loaded() {
  await screen.findByRole("heading", { name: "Production Quality Report" });
}

describe("InspectionDetailPage", () => {
  beforeAll(() => {
    // jsdom does not implement these; the page revokes its image object URL on unmount.
    URL.revokeObjectURL = vi.fn();
  });

  beforeEach(() => {
    getInspection.mockReset().mockResolvedValue(AI_INSPECTION);
    getInspectionReport.mockReset().mockResolvedValue(report());
    getInspectionImageObjectUrl.mockReset().mockResolvedValue("blob:mock");
  });

  describe("production quality report", () => {
    it("shows the overall result, the report's completeness, and the summary", async () => {
      renderPage();
      await loaded();

      await waitFor(() => expect(reportCard().getByText("Complete")).toBeInTheDocument());
      expect(reportCard().getByText("Report status")).toBeInTheDocument();
      expect(reportCard().getByText("Overall Result")).toBeInTheDocument();
      expect(reportCard().getByText("FAIL")).toBeInTheDocument();
      expect(reportCard().getByText("Ground truth and AI agree the part is defective.")).toBeInTheDocument();
    });

    it("shows a partial report as Partial, without touching the overall result", async () => {
      getInspectionReport.mockResolvedValue(
        report({ overall_result: "NOT_ASSESSED", report_status: "PARTIAL", summary: "Quality assessment has not been run for this inspection." })
      );
      renderPage();
      await loaded();

      await waitFor(() => expect(reportCard().getByText("Partial")).toBeInTheDocument());
      expect(reportCard().getByText("NOT ASSESSED")).toBeInTheDocument();
    });

    it("states that report status is not a quality result", async () => {
      renderPage();
      await loaded();

      expect(reportCard().getByText(/it is not a quality result/)).toBeInTheDocument();
    });

    it("shows a failed report as an error with Retry - never as an endless loading placeholder", async () => {
      getInspectionReport.mockRejectedValue(new Error("Report request failed"));
      renderPage();
      await loaded();

      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("Report request failed");
      expect(reportCard().queryByText("Report status")).not.toBeInTheDocument();
    });

    it("still renders the rest of the page when only the report fails", async () => {
      getInspectionReport.mockRejectedValue(new Error("Report request failed"));
      renderPage();
      await screen.findByRole("alert");

      expect(screen.getByRole("heading", { name: "Quality Assessment" })).toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "AI Prediction" })).toBeInTheDocument();
      expect(screen.getByText("#5")).toBeInTheDocument();
    });

    it("Retry requests the report again and shows it once it loads", async () => {
      getInspectionReport
        .mockRejectedValueOnce(new Error("Report request failed"))
        .mockResolvedValueOnce(report());
      renderPage();
      const alert = await screen.findByRole("alert");

      fireEvent.click(within(alert).getByRole("button", { name: "Retry" }));

      await waitFor(() => expect(reportCard().getByText("Complete")).toBeInTheDocument());
      expect(getInspectionReport).toHaveBeenCalledTimes(2);
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("shows neither an error nor a result while the report is still loading", async () => {
      getInspectionReport.mockReturnValue(new Promise(() => {}));
      renderPage();
      await loaded();

      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(reportCard().queryByText("Report status")).not.toBeInTheDocument();
    });
  });

  describe("timings", () => {
    it("shows the recorded processing and AI analysis times", async () => {
      renderPage();
      await loaded();

      await screen.findByText("845 ms");
      expect(screen.getByText("Processing time")).toBeInTheDocument();
      expect(screen.getByText("AI analysis time")).toBeInTheDocument();
      expect(screen.getByText("12.3 s")).toBeInTheDocument();
    });

    it("says 'Not recorded' - not 0 - for an inspection saved before timing existed", async () => {
      getInspection.mockResolvedValue({ ...AI_INSPECTION, processing_time_ms: null, ai_inference_time_ms: null });
      renderPage();
      await loaded();

      await waitFor(() => expect(screen.getAllByText("Not recorded")).toHaveLength(2));
      expect(screen.queryByText("0 ms")).not.toBeInTheDocument();
    });

    it("shows no AI analysis time for an inspection the AI never analyzed", async () => {
      getInspection.mockResolvedValue(UPLOAD_INSPECTION);
      renderPage(6);
      await loaded();

      await screen.findByText("210 ms");
      expect(screen.queryByText("AI analysis time")).not.toBeInTheDocument();
      expect(screen.getByText("Not yet analyzed")).toBeInTheDocument();
    });
  });

  describe("evidence sections stay distinct and honest", () => {
    it("keeps the exact 'not assessed' wording when severity evidence is missing", async () => {
      renderPage();
      await loaded();

      expect(await screen.findByText("Not assessed - insufficient evidence available")).toBeInTheDocument();
    });

    it("still shows the dataset reference for an MVTec inspection and none for an upload", async () => {
      const { unmount } = renderPage();
      await loaded();
      expect(await screen.findByRole("heading", { name: "Dataset reference" })).toBeInTheDocument();
      unmount();

      getInspection.mockResolvedValue(UPLOAD_INSPECTION);
      renderPage(6);
      await loaded();
      expect(screen.queryByRole("heading", { name: "Dataset reference" })).not.toBeInTheDocument();
    });
  });
});
