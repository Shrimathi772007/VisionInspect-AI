import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../../api/inspections", () => ({ batchUploadInspections: vi.fn() }));

import { batchUploadInspections } from "../../api/inspections";
import { BatchUploadPanel } from "./BatchUploadPanel";

function makeFile(name, size = 1024, type = "image/png") {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function renderPanel(productId = "4") {
  return render(
    <MemoryRouter>
      <BatchUploadPanel productId={productId} />
    </MemoryRouter>
  );
}

const addFiles = (files) => fireEvent.change(screen.getByTestId("batch-file-input"), { target: { files } });
const uploadButton = () => screen.getByRole("button", { name: "Upload batch" });

const inspection = (id, overrides = {}) => ({
  id,
  quality_decision: "PASS",
  ai_prediction: "good",
  ai_confidence: 0.9612,
  review_required: false,
  ...overrides,
});

describe("BatchUploadPanel", () => {
  beforeEach(() => {
    batchUploadInspections.mockReset();
  });

  it("lists the chosen files with their sizes and lets one be removed", () => {
    renderPanel();
    addFiles([makeFile("a.png", 2048), makeFile("b.jpg", 3 * 1024 * 1024)]);
    const list = within(screen.getByRole("list", { name: "Selected files" }));
    expect(list.getByText("a.png")).toBeInTheDocument();
    expect(list.getByText("2.0 KB")).toBeInTheDocument();
    expect(list.getByText("3.0 MB")).toBeInTheDocument();
    expect(screen.getByText(/2 files/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Remove a.png" }));
    expect(list.queryByText("a.png")).not.toBeInTheDocument();
    expect(uploadButton()).toBeEnabled();
  });

  it("rejects more than 20 files", () => {
    renderPanel();
    addFiles(Array.from({ length: 21 }, (_, i) => makeFile(`${i}.png`)));
    expect(screen.getByRole("alert")).toHaveTextContent("Too many files: 21 selected, at most 20 per batch.");
    expect(uploadButton()).toBeDisabled();
  });

  it("rejects a batch over 50 MB in total", () => {
    renderPanel();
    addFiles(Array.from({ length: 6 }, (_, i) => makeFile(`${i}.png`, 9 * 1024 * 1024)));
    expect(screen.getByRole("alert")).toHaveTextContent("exceeds the 50 MB batch limit");
    expect(uploadButton()).toBeDisabled();
  });

  it("rejects unsupported extensions and files over 10 MB", () => {
    renderPanel();
    addFiles([makeFile("notes.gif", 10, "image/gif"), makeFile("big.png", 11 * 1024 * 1024), makeFile("ok.jpeg")]);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Unsupported file type (JPEG or PNG only): notes.gif.");
    expect(alert).toHaveTextContent("Larger than 10 MB: big.png.");
    expect(uploadButton()).toBeDisabled();
  });

  it("is disabled without a product", () => {
    renderPanel("");
    addFiles([makeFile("a.png")]);
    expect(uploadButton()).toBeDisabled();
  });

  it("shows progress, then a results table with successes, failures, links and the summary", async () => {
    let resolve;
    batchUploadInspections.mockReturnValue(new Promise((r) => (resolve = r)));
    renderPanel("4");
    const files = [makeFile("a.png"), makeFile("b.png"), makeFile("c.png")];
    addFiles(files);
    fireEvent.click(uploadButton());

    expect(await screen.findByText("Uploading and analysing 3 images…")).toBeInTheDocument();
    expect(batchUploadInspections).toHaveBeenCalledWith({ productId: "4", files });

    resolve({
      total: 3,
      succeeded: 2,
      failed: 1,
      items: [
        { filename: "a.png", inspection: inspection(11), error: null },
        { filename: "b.png", inspection: null, error: "Uploaded file is not a valid JPEG or PNG image." },
        {
          filename: "c.png",
          inspection: inspection(12, { quality_decision: "MANUAL_REVIEW", ai_prediction: "defective", review_required: true }),
          error: null,
        },
      ],
    });

    expect(await screen.findByText("2 of 3 succeeded")).toBeInTheDocument();
    const rows = screen.getAllByTestId("batch-result-row");
    expect(within(rows[0]).getByText("PASS")).toBeInTheDocument();
    expect(within(rows[0]).getByText("AI: Good")).toBeInTheDocument();
    expect(within(rows[0]).getByText("96.1%")).toBeInTheDocument();
    expect(within(rows[0]).getByRole("link", { name: "View #11" })).toHaveAttribute("href", "/inspections/11");
    expect(within(rows[1]).getByText("Failed")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Uploaded file is not a valid JPEG or PNG image.")).toBeInTheDocument();
    expect(within(rows[1]).queryByRole("link")).not.toBeInTheDocument();
    expect(within(rows[2]).getByText("MANUAL REVIEW")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Review")).toBeInTheDocument();
    expect(within(rows[2]).getByRole("link", { name: "View #12" })).toHaveAttribute("href", "/inspections/12");

    fireEvent.click(screen.getByRole("button", { name: "Upload another batch" }));
    expect(screen.queryByText("2 of 3 succeeded")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/yolo/i);
  });

  it("shows a request error and keeps the files", async () => {
    batchUploadInspections.mockRejectedValue(new Error("network"));
    renderPanel();
    addFiles([makeFile("a.png")]);
    fireEvent.click(uploadButton());
    expect(await screen.findByText("Batch upload failed. Please try again.")).toBeInTheDocument();
    expect(screen.getByText("a.png")).toBeInTheDocument();
  });

  it("renders an unknown decision safely", async () => {
    batchUploadInspections.mockResolvedValue({
      total: 1,
      succeeded: 1,
      failed: 0,
      items: [{ filename: "a.png", inspection: inspection(3, { quality_decision: "NEW_VALUE", ai_confidence: null }), error: null }],
    });
    renderPanel();
    addFiles([makeFile("a.png")]);
    fireEvent.click(uploadButton());
    expect(await screen.findByText("NEW_VALUE")).toBeInTheDocument();
  });
});
