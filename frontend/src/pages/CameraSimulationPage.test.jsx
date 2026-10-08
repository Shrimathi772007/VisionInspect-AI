import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../hooks/useProducts", () => ({
  useProducts: () => ({
    products: [{ id: 1, product_name: "Tile A", product_code: "TILE-1", category: "tile" }],
    isLoading: false,
  }),
}));
const { categoriesState } = vi.hoisted(() => ({ categoriesState: { current: null } }));
const DEFAULT_CATEGORIES = { categories: ["bottle", "tile"], isLoading: false, error: null, refetch: () => {} };
vi.mock("../hooks/useDatasetCategories", () => ({
  useDatasetCategories: () => categoriesState.current,
}));
vi.mock("../api/dataset", () => ({
  getDatasetCategoryDetail: vi.fn(),
  listDatasetImages: vi.fn(),
  importDatasetInspection: vi.fn(),
}));

import { ApiError } from "../api/client";
import { getDatasetCategoryDetail, importDatasetInspection, listDatasetImages } from "../api/dataset";
import { CameraSimulationPage } from "./CameraSimulationPage";

const FILES = { crack: ["001.png", "000.png"], good: ["002.png", "000.png", "001.png"] };
const BANNER = "Images come from the MVTec AD dataset. Results here are demonstration data, not accuracy measurements.";

let importCount = 0;

function renderPage() {
  return render(
    <MemoryRouter>
      <CameraSimulationPage />
    </MemoryRouter>
  );
}

// Flush resolved promises (and the React updates they cause) without moving the clock.
const flush = () => act(async () => {});
const advance = (ms) => act(async () => {
  await vi.advanceTimersByTimeAsync(ms);
});

async function configure({ category = "tile", interval = "1", frames = "10", defectType } = {}) {
  fireEvent.change(screen.getByLabelText("Product"), { target: { value: "1" } });
  fireEvent.change(screen.getByLabelText("Dataset category"), { target: { value: category } });
  await flush();
  if (defectType) fireEvent.change(screen.getByLabelText("Defect type"), { target: { value: defectType } });
  fireEvent.change(screen.getByLabelText("Interval (1-10 s)"), { target: { value: interval } });
  fireEvent.change(screen.getByLabelText("Max frames (1-50)"), { target: { value: frames } });
}

async function start() {
  fireEvent.click(screen.getByRole("button", { name: "Start simulated camera" }));
  await flush();
}

const importedNames = () => importDatasetInspection.mock.calls.map(([args]) => `${args.defectType}/${args.filename}`);

describe("CameraSimulationPage", () => {
  beforeEach(() => {
    categoriesState.current = DEFAULT_CATEGORIES;
    vi.useFakeTimers();
    importCount = 0;
    getDatasetCategoryDetail.mockReset().mockImplementation(async (category) => ({
      category,
      splits: [
        { split: "test", defect_types: [{ defect_type: "good", count: 3 }, { defect_type: "crack", count: 2 }] },
        { split: "train", defect_types: [{ defect_type: "good", count: 3 }] },
      ],
    }));
    listDatasetImages.mockReset().mockImplementation(async ({ category, split, defectType }) => ({
      category, split, defect_type: defectType, filenames: FILES[defectType] ?? [],
    }));
    importDatasetInspection.mockReset().mockImplementation(async ({ defectType }) => {
      importCount += 1;
      return {
        id: 100 + importCount,
        status: defectType === "good" ? "good" : "defective",
        ai_prediction: "defective",
        ai_confidence: 0.9,
        quality_decision: "FAIL",
        review_required: false,
      };
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the simulation banner", () => {
    renderPage();
    expect(screen.getByText("Camera simulation", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByText(BANNER)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/yolo/i);
  });

  it("imports the first image only after one interval", async () => {
    renderPage();
    await configure({ interval: "3" });
    await start();
    expect(importDatasetInspection).not.toHaveBeenCalled();
    await advance(2999);
    expect(importDatasetInspection).not.toHaveBeenCalled();
    await advance(1);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
    expect(importDatasetInspection).toHaveBeenCalledWith({
      productId: "1", category: "tile", split: "test", defectType: "crack", filename: "000.png",
    });
    expect(screen.getByText("Frame 1 of 10")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View #101" })).toHaveAttribute("href", "/inspections/101");
  });

  it("follows a deterministic order, cycles, and stops at max frames", async () => {
    renderPage();
    await configure({ interval: "1", frames: "7" });
    await start();
    for (let i = 0; i < 7; i += 1) await advance(1000);
    expect(importedNames()).toEqual([
      "crack/000.png", "good/000.png", "crack/001.png", "good/001.png", "good/002.png", "crack/000.png", "good/000.png",
    ]);
    expect(screen.getByText("Finished: 7 frames imported.")).toBeInTheDocument();
    expect(screen.getByText("Frame 7 of 7")).toBeInTheDocument();
    expect(screen.getAllByTestId("camera-frame")).toHaveLength(7);

    await advance(10000);
    expect(importDatasetInspection).toHaveBeenCalledTimes(7);
    expect(screen.getByRole("button", { name: "Start simulated camera" })).toBeInTheDocument();
  });

  it("uses only the chosen defect type", async () => {
    renderPage();
    await configure({ interval: "1", frames: "3", defectType: "good" });
    await start();
    for (let i = 0; i < 3; i += 1) await advance(1000);
    expect(importedNames()).toEqual(["good/000.png", "good/001.png", "good/002.png"]);
  });

  it("Stop stops the camera", async () => {
    renderPage();
    await configure({ interval: "1" });
    await start();
    await advance(1000);
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await advance(10000);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Stopped.")).toBeInTheDocument();
  });

  it("an API error stops the camera with a message", async () => {
    importDatasetInspection.mockRejectedValueOnce(new ApiError(500, "Import failed"));
    renderPage();
    await configure({ interval: "1" });
    await start();
    await advance(1000);
    expect(screen.getByText("Camera stopped: Import failed")).toBeInTheDocument();
    await advance(10000);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Start simulated camera" })).toBeInTheDocument();
  });

  it("unmounting clears the timer", async () => {
    const { unmount } = renderPage();
    await configure({ interval: "2" });
    await start();
    expect(vi.getTimerCount()).toBe(1);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
    await advance(10000);
    expect(importDatasetInspection).not.toHaveBeenCalled();
  });

  it("never runs two requests at once", async () => {
    let release;
    importDatasetInspection.mockImplementationOnce(
      () => new Promise((resolve) => (release = () => resolve({ id: 7, status: "good", quality_decision: "PASS" })))
    );
    renderPage();
    await configure({ interval: "1" });
    await start();
    await advance(1000);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
    await advance(5000); // the request is still pending: nothing else is sent
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);

    await act(async () => release());
    await advance(999);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
    await advance(1);
    expect(importDatasetInspection).toHaveBeenCalledTimes(2);
  });

  it("clamps the interval and frame limits", async () => {
    renderPage();
    await configure({ interval: "0", frames: "99" });
    await start();
    expect(screen.getByLabelText("Interval (1-10 s)")).toHaveValue(1);
    expect(screen.getByLabelText("Max frames (1-50)")).toHaveValue(50);
    expect(screen.getByText("Frame 0 of 50")).toBeInTheDocument();
    await advance(1000);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
  });

  it("clamps an interval above 10 s and keeps the controls locked while running", async () => {
    renderPage();
    await configure({ interval: "60", frames: "1" });
    await start();
    expect(screen.getByLabelText("Interval (1-10 s)")).toHaveValue(10);
    expect(screen.getByLabelText("Product")).toBeDisabled();
    await advance(9999);
    expect(importDatasetInspection).not.toHaveBeenCalled();
    await advance(1);
    expect(importDatasetInspection).toHaveBeenCalledTimes(1);
  });

  it("warns when the product category differs from the dataset category", async () => {
    renderPage();
    await configure({ category: "bottle" });
    expect(screen.getByText(/Product category differs from the dataset category/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Dataset category"), { target: { value: "tile" } });
    await flush();
    expect(screen.queryByText(/Product category differs from the dataset category/)).not.toBeInTheDocument();
  });

  it("reports an empty selection instead of starting", async () => {
    listDatasetImages.mockImplementation(async () => ({ filenames: [] }));
    renderPage();
    await configure();
    await start();
    expect(screen.getByText("No images found for this category, split and defect type.")).toBeInTheDocument();
    expect(vi.getTimerCount()).toBe(0);
  });

  describe("without dataset categories", () => {
    const NO_CATEGORIES = "No dataset categories found. Check that the dataset folder is mounted.";

    it("explains an empty category list and disables the dropdown", () => {
      categoriesState.current = { ...DEFAULT_CATEGORIES, categories: [] };
      renderPage();
      expect(screen.getByRole("alert")).toHaveTextContent(NO_CATEGORIES);
      expect(screen.getByLabelText("Dataset category")).toBeDisabled();
      expect(screen.getByRole("button", { name: "Start simulated camera" })).toBeDisabled();
    });

    it("shows the same message when the categories fail to load, with a retry", () => {
      const refetch = vi.fn();
      categoriesState.current = { ...DEFAULT_CATEGORIES, categories: [], error: new ApiError(500, "boom"), refetch };
      renderPage();
      expect(screen.getByRole("alert")).toHaveTextContent(NO_CATEGORIES);
      expect(screen.queryByText(/boom/)).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      expect(refetch).toHaveBeenCalledTimes(1);
    });

    it("shows no message while loading or when categories exist", () => {
      categoriesState.current = { ...DEFAULT_CATEGORIES, categories: [], isLoading: true };
      const { unmount } = renderPage();
      expect(screen.queryByText(NO_CATEGORIES)).not.toBeInTheDocument();
      unmount();
      categoriesState.current = DEFAULT_CATEGORIES;
      renderPage();
      expect(screen.queryByText(NO_CATEGORIES)).not.toBeInTheDocument();
    });
  });
});
