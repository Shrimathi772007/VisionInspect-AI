import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SeverityCard } from "./SeverityCard";

const DEFECT_FORM_NOTE = "Shape-based form derived from the anomaly region; not a trained defect-type classifier.";
const SEVERITY_NOTE =
  "Severity is rule-based: size, location (centre-weighted proxy), form and confidence. Region size is approximate.";

const SCORED = {
  status: "pending",
  severity_score: 70.5,
  severity_level: "High",
  quality_risk: "High Risk",
  severity_action: "Repair or rework recommended",
  defect_form: "localized_patch",
  defect_form_label: "Localized patch",
};

describe("SeverityCard", () => {
  it("shows the score, level, risk, recommended action, defect form and both notes", () => {
    render(<SeverityCard inspection={SCORED} />);
    expect(screen.getByText("71 / 100")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
    expect(screen.getByText("High Risk")).toBeInTheDocument();
    expect(screen.getByText("Recommended action")).toBeInTheDocument();
    expect(screen.getByText("Repair or rework recommended")).toBeInTheDocument();
    expect(screen.getByText("Localized patch")).toBeInTheDocument();
    expect(screen.getByText(DEFECT_FORM_NOTE)).toBeInTheDocument();
    expect(screen.getByText(SEVERITY_NOTE)).toBeInTheDocument();
  });

  it.each([
    ["Low", "success"],
    ["Medium", "info"],
    ["High", "warning"],
    ["Critical", "danger"],
    ["Extreme", "neutral"],
  ])("renders level %s with tone %s (unknown levels render as-is)", (level, tone) => {
    render(<SeverityCard inspection={{ ...SCORED, severity_level: level }} />);
    expect(screen.getByText(level).className).toMatch(new RegExp(tone));
  });

  it("explains a missing severity for an upload", () => {
    render(<SeverityCard inspection={{ status: "pending", severity_score: null, severity_level: null }} />);
    expect(screen.getByText("No severity: the inspection is good or has no localized region.")).toBeInTheDocument();
    expect(screen.queryByText(SEVERITY_NOTE)).not.toBeInTheDocument();
  });

  it.each(["good", "defective"])("says the AI found no defect for an AI-good import (ground truth %s)", (status) => {
    render(<SeverityCard inspection={{ status, ai_prediction: "good", severity_score: null }} />);
    expect(screen.getByText("No defect detected by the AI, so severity does not apply.")).toBeInTheDocument();
    expect(screen.queryByText("Not assessed - insufficient evidence available")).not.toBeInTheDocument();
  });

  it("keeps the upload wording for an AI-good upload", () => {
    render(<SeverityCard inspection={{ status: "pending", ai_prediction: "good", severity_score: null }} />);
    expect(screen.getByText("No severity: the inspection is good or has no localized region.")).toBeInTheDocument();
  });

  it.each([null, "defective"])("keeps the ground-truth wording for an import without severity (AI %s)", (ai) => {
    render(<SeverityCard inspection={{ status: "defective", ai_prediction: ai, severity_score: null }} />);
    expect(screen.getByText("Not assessed - insufficient evidence available")).toBeInTheDocument();
  });

  it("shows the severity_v1 score for an AI-defective import", () => {
    render(<SeverityCard inspection={{ ...SCORED, status: "good", ai_prediction: "defective" }} />);
    expect(screen.getByText("71 / 100")).toBeInTheDocument();
    expect(screen.getByText(SEVERITY_NOTE)).toBeInTheDocument();
  });

  it("omits the defect form rows when there is no form", () => {
    render(<SeverityCard inspection={{ ...SCORED, defect_form: null, defect_form_label: null, severity_action: null }} />);
    expect(screen.queryByText(DEFECT_FORM_NOTE)).not.toBeInTheDocument();
    expect(screen.queryByText("Recommended action")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/yolo/i);
  });
});
