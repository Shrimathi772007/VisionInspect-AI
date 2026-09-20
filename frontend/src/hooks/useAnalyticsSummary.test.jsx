import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

vi.mock("../api/analytics", () => ({ getInspectionAnalyticsSummary: vi.fn() }));

import { getInspectionAnalyticsSummary } from "../api/analytics";
import { useAnalyticsSummary } from "./useAnalyticsSummary";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("useAnalyticsSummary", () => {
  beforeEach(() => {
    getInspectionAnalyticsSummary.mockReset();
  });

  it("starts in a loading state and then exposes the loaded data", async () => {
    getInspectionAnalyticsSummary.mockResolvedValue({ marker: "loaded" });

    const { result } = renderHook(() => useAnalyticsSummary({ days: 14 }));

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeNull();

    await waitFor(() => expect(result.current.data).toEqual({ marker: "loaded" }));
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("requests the selected number of days", async () => {
    getInspectionAnalyticsSummary.mockResolvedValue({});

    renderHook(() => useAnalyticsSummary({ days: 30 }));

    await waitFor(() => expect(getInspectionAnalyticsSummary).toHaveBeenCalledWith({ days: 30 }));
  });

  it("keeps the previous data on screen while a new range is loading", async () => {
    const second = deferred();
    getInspectionAnalyticsSummary
      .mockResolvedValueOnce({ marker: "14-day data" })
      .mockReturnValueOnce(second.promise);

    const { result, rerender } = renderHook(({ days }) => useAnalyticsSummary({ days }), {
      initialProps: { days: 14 },
    });
    await waitFor(() => expect(result.current.data).toEqual({ marker: "14-day data" }));

    rerender({ days: 7 });

    // Old data is still there (no skeleton flash) and the hook reports it is refreshing.
    expect(result.current.data).toEqual({ marker: "14-day data" });
    expect(result.current.isRefreshing).toBe(true);
    expect(result.current.isLoading).toBe(false);

    await act(async () => {
      second.resolve({ marker: "7-day data" });
    });
    await waitFor(() => expect(result.current.data).toEqual({ marker: "7-day data" }));
    expect(result.current.isRefreshing).toBe(false);
  });

  it("ignores a slow response that arrives after the range has changed", async () => {
    const slowFirst = deferred();
    getInspectionAnalyticsSummary
      .mockReturnValueOnce(slowFirst.promise)
      .mockResolvedValueOnce({ marker: "7-day data" });

    const { result, rerender } = renderHook(({ days }) => useAnalyticsSummary({ days }), {
      initialProps: { days: 14 },
    });
    rerender({ days: 7 });
    await waitFor(() => expect(result.current.data).toEqual({ marker: "7-day data" }));

    await act(async () => {
      slowFirst.resolve({ marker: "stale 14-day data" });
    });

    expect(result.current.data).toEqual({ marker: "7-day data" });
  });

  it("exposes an error, with no data, when the request fails", async () => {
    const failure = new Error("Analytics exploded");
    getInspectionAnalyticsSummary.mockRejectedValue(failure);

    const { result } = renderHook(() => useAnalyticsSummary({ days: 14 }));

    await waitFor(() => expect(result.current.error).toBe(failure));
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it("refetch clears the error and loads again", async () => {
    getInspectionAnalyticsSummary
      .mockRejectedValueOnce(new Error("first attempt failed"))
      .mockResolvedValueOnce({ marker: "recovered" });

    const { result } = renderHook(() => useAnalyticsSummary({ days: 14 }));
    await waitFor(() => expect(result.current.error).not.toBeNull());

    act(() => {
      result.current.refetch();
    });

    await waitFor(() => expect(result.current.data).toEqual({ marker: "recovered" }));
    expect(result.current.error).toBeNull();
    expect(getInspectionAnalyticsSummary).toHaveBeenCalledTimes(2);
  });
});
