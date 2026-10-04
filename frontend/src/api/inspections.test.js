import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./client", () => ({ apiFetch: vi.fn() }));

import { apiFetch } from "./client";
import { batchUploadInspections, getInspectionEnhancedBlob, getInspectionImageQuality } from "./inspections";

describe("inspections API (batch and enhancement)", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockResolvedValue({});
  });

  it("sends a batch as product_id plus repeated 'files' fields", () => {
    const files = [new File(["a"], "a.png"), new File(["b"], "b.jpg")];
    batchUploadInspections({ productId: 7, files });

    const [path, options] = apiFetch.mock.calls[0];
    expect(path).toBe("/inspections/batch");
    expect(options.method).toBe("POST");
    expect(options.formData.get("product_id")).toBe("7");
    expect(options.formData.getAll("files").map((file) => file.name)).toEqual(["a.png", "b.jpg"]);
  });

  it("fetches the enhanced preview as a blob and the metrics as JSON", () => {
    getInspectionEnhancedBlob(5);
    getInspectionImageQuality(5);
    expect(apiFetch).toHaveBeenCalledWith("/inspections/5/enhanced", { responseType: "blob" });
    expect(apiFetch).toHaveBeenCalledWith("/inspections/5/image-quality");
  });
});
