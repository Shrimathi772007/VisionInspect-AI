import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

vi.mock("../api/analytics", () => ({ getCategoryAnalytics: vi.fn() }));

import { getCategoryAnalytics } from "../api/analytics";
import { useCategoryAnalytics } from "./useCategoryAnalytics";

function deferred() {
  let resolve;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("useCategoryAnalytics", () => {
  beforeEach(() => {
    getCategoryAnalytics.mockReset();
  });

  it("loads the selected window and exposes the data", async () => {
    getCategoryAnalytics.mockResolvedValue({ window_days: 30, categories: [] });

    const { result } = renderHook(() => useCategoryAnalytics({ days: 30 }));
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.data).toEqual({ window_days: 30, categories: [] }));
    expect(getCategoryAnalytics).toHaveBeenCalledWith({ days: 30 });
    expect(result.current.error).toBeNull();
  });

  it("keeps the previous data while a new range loads and ignores a stale response", async () => {
    const slow = deferred();
    getCategoryAnalytics.mockResolvedValueOnce({ marker: "14" }).mockReturnValueOnce(slow.promise);

    const { result, rerender } = renderHook(({ days }) => useCategoryAnalytics({ days }), {
      initialProps: { days: 14 },
    });
    await waitFor(() => expect(result.current.data).toEqual({ marker: "14" }));

    getCategoryAnalytics.mockResolvedValueOnce({ marker: "30" });
    rerender({ days: 7 });
    expect(result.current.isRefreshing).toBe(true);
    expect(result.current.data).toEqual({ marker: "14" });

    rerender({ days: 30 });
    await waitFor(() => expect(result.current.data).toEqual({ marker: "30" }));
    await act(async () => {
      slow.resolve({ marker: "stale 7" });
    });
    expect(result.current.data).toEqual({ marker: "30" });
  });

  it("exposes an error and recovers on refetch", async () => {
    const failure = new Error("by-category failed");
    getCategoryAnalytics.mockRejectedValueOnce(failure).mockResolvedValueOnce({ marker: "ok" });

    const { result } = renderHook(() => useCategoryAnalytics({ days: 14 }));
    await waitFor(() => expect(result.current.error).toBe(failure));
    expect(result.current.data).toBeNull();

    act(() => {
      result.current.refetch();
    });
    await waitFor(() => expect(result.current.data).toEqual({ marker: "ok" }));
    expect(result.current.error).toBeNull();
  });
});
