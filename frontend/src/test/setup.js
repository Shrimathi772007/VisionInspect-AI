import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Testing Library only auto-cleans when the test framework exposes globals; we import
// explicitly instead, so unmount rendered trees after every test ourselves.
afterEach(() => {
  cleanup();
});
