"""Assemble docs/PROJECT_DOCUMENTATION.md from the milestone files and the shared documents.

    python docs/build_full_doc.py

Inputs (all must exist, or the script stops with an error):
    docs/milestones/MILESTONE_1.md ... MILESTONE_4.md   the single source for each milestone
    docs/ARCHITECTURE.md, docs/ML_METHODOLOGY.md         embedded as chapters
    README.md, docs/DEPLOY.md, LICENSE.txt               linked from the short operations sections

The introduction, the requirements traceability appendix and the stack-as-built table are written below. The output
is deterministic: no timestamps, fixed input order, "\n" line endings. Never edit the generated file by hand; change
a milestone file or this script and run it again.

Embedded files are adjusted in two ways: their headings move one level down (outside code fences), and their
relative links are rewritten so they resolve from docs/. A link to an anchor in the same file becomes a link to that
file and anchor, so anchors never collide between chapters. A relative link whose target does not exist stops the
build.
"""

import posixpath
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT = "docs/PROJECT_DOCUMENTATION.md"
MILESTONES = [f"docs/milestones/MILESTONE_{n}.md" for n in (1, 2, 3, 4)]
SHARED = ["docs/ARCHITECTURE.md", "docs/ML_METHODOLOGY.md"]
LINKED = ["README.md", "docs/DEPLOY.md", "LICENSE.txt"]

LINK = re.compile(r"\]\(([^)\s]+)\)")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6}) ")


def fail(message: str) -> None:
    print(f"build_full_doc.py: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(relative: str) -> str:
    path = REPO / relative
    if not path.is_file():
        fail(f"missing input file: {relative}")
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def slug(heading: str) -> str:
    """GitHub-style anchor of a heading (without the duplicate-suffix rule)."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def demote(text: str) -> str:
    """Move every Markdown heading one level down, leaving code fences untouched."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and HEADING.match(line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def rewrite_links(text: str, source: str) -> str:
    """Rewrite relative links of `source` (repo-relative) so they resolve from the output's folder."""
    source_dir = posixpath.dirname(source)
    output_dir = posixpath.dirname(OUTPUT)

    def replace(match: re.Match) -> str:
        target = match.group(1)
        if re.match(r"^[a-z]+:", target):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(source_dir, path)) if path else source
        if not (REPO / resolved).exists():
            fail(f"{source}: link target does not exist: {target}")
        new = posixpath.relpath(resolved, output_dir)
        if path.endswith("/") and not new.endswith("/"):
            new += "/"
        return f"]({new}#{anchor})" if anchor else f"]({new})"

    out, in_fence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            line = LINK.sub(replace, line)
        out.append(line)
    return "\n".join(out)


def embed(relative: str) -> tuple[str, str]:
    """(first heading text, adjusted body) of one input file."""
    text = read(relative).strip("\n")
    first = text.split("\n", 1)[0]
    if not first.startswith("# "):
        fail(f"{relative}: must start with a level-1 heading")
    return first[2:].strip(), demote(rewrite_links(text, relative))


INTRODUCTION = """\
## Introduction

VisionInspect AI is a web application for visual quality inspection of manufactured parts. Images are uploaded or
imported from the MVTec AD dataset, scored by an anomaly-detection model chosen for their MVTec category, localized
with a heatmap and boxes, given a rule-based severity and a quality decision, and summarised on a dashboard. The
system was built in four milestones between 27 August and 17 October 2026. This document collects the four
milestone reports, a traceability appendix against the project's module and metric lists, a comparison of the
planned and the actual technology stack, and the architecture and machine-learning documents.

Each milestone report lists the commits that belong to it by content, with their dates. Work that completed a
milestone's task after that milestone's own dates is listed inside that milestone as delivered in a later
iteration. All dates come from the Git history, all metrics from the code and the stored result files.

| Milestone | Scope | Commit dates of the milestone's own work |
|---|---|---|
| 1 | Setup, authentication, upload, dataset integration, dashboard | 27 August to 5 September 2026 |
| 2 | Preprocessing, image quality, detection models, predictions, monitoring | 5 to 24 September 2026 |
| 3 | Defect category, severity, quality decision, reports, analytics, trends | 12 to 14 September 2026 |
| 4 | Validation, optimization, deployment, documentation | 20 September to 17 October 2026 |

The live deployment runs at http://13.206.115.50 (plain HTTP, one AWS EC2 VM). Setup, testing and deployment are
described in [README.md](../README.md) and [DEPLOY.md](DEPLOY.md).
"""

TRACEABILITY = """\
## Appendix A: Requirements traceability

### Modules

Status as of 17 October 2026. DONE means implemented and covered by tests or stored results; PARTIAL means part of
the requirement is implemented and the rest is stated; a deviation means the requirement is met in a different way
than specified.

| Module | Requirement | Status | Evidence |
|---|---|---|---|
| User management | Quality engineer accounts | DONE | `backend/scripts/create_quality_engineer.py`; promotion through `PATCH /users/{user_id}/role` |
| User management | Factory supervisor accounts | DONE | `POST /auth/register` always creates a factory supervisor |
| User management | Authentication and authorization | DONE | JWT and bcrypt in `backend/app/auth/`; `require_role` on every protected route; failed-login limiter |
| User management | Role management | DONE | Users page; the last quality engineer cannot be demoted (`backend/app/users/service.py`) |
| Image acquisition | Product image upload | DONE | `POST /inspections/upload` |
| Image acquisition | Camera integration simulation | DONE | Camera simulation page: timed imports of labelled dataset images; not a live camera |
| Image acquisition | Batch image processing | DONE | `POST /inspections/batch`, up to 20 files and 50 MB |
| Image acquisition | Image validation | DONE | Extension, streamed size limit and Pillow format check in `backend/app/inspections/storage.py` |
| Image processing | Image preprocessing | DONE | `backend/app/ai/preprocessing/pipeline.py`; served models use `patchcore_preprocess.py` |
| Image processing | Noise removal | PARTIAL | Non-local-means denoising in the enhancement preview (`backend/app/ai/analytics/enhancement.py`); computed per request, not stored and not used by the models |
| Image processing | Image enhancement | PARTIAL | CLAHE contrast in the same preview, with the same limits |
| Image processing | Feature extraction | DONE | Frozen ImageNet ResNet-18 and Wide ResNet-50-2 patch features (layers 2 and 3) |
| Defect detection | Defect identification | DONE | Image-level prediction against a locked threshold for all 15 categories (`backend/app/ai/inference/serving.py`) |
| Defect detection | Anomaly detection | DONE | ResNet-18 patch models and WRN-50 PatchCore, trained on defect-free images only |
| Defect detection | Object detection | Deviation | No object detector or segmentation model is used; boxes are derived from the anomaly map |
| Defect detection | Defect localization | DONE | Heatmap and up to 5 boxes per image (`backend/app/ai/inference/localization.py`); box AP@0.5 mean 0.200, pixel AUROC mean 0.9759 |
| Defect classification | Defect type classification | PARTIAL | Imports carry the ground-truth defect folder name; `defect_form_v1` describes the shape of the flagged region. No model predicts the defect type. |
| Defect classification | Severity scoring | DONE, with two deviations | Weights 30/25/25/20 and bands Critical 80-100, High 60-79, Medium 40-59, Low 0-39 in `backend/app/inspections/severity.py`. Deviation: the defect-type factor is filled by `defect_form_v1` (`backend/app/inspections/defect_form.py`) because no defect-type classifier exists. Deviation: the location factor is a centre-weighted proxy for critical area (`location_score_from_centroid`). |
| Defect classification | Defect categorization | PARTIAL | `defect_category` from the dataset folder for imports; uploads have none |
| Defect classification | Confidence analysis | DONE | Margin confidence; high at 0.95 or more, manual review below 0.70 (`reliability_level`, `review_rule` in `localization.py`) |
| Quality control | Pass/fail decision | DONE | PASS, FAIL, MANUAL_REVIEW or NOT_ASSESSED (`backend/app/inspections/quality.py`) |
| Quality control | Inspection reporting | DONE | `GET /inspections/{id}/report` |
| Quality control | Quality recommendations | DONE | Recommendation text per decision in `quality.py` |
| Quality control | Production monitoring | DONE | Dashboard and `GET /inspections/analytics/summary` |
| Analytics dashboard | Defect trend analysis | DONE | Daily series, category trends and window comparison (`trend_monitoring`) |
| Analytics dashboard | Production quality reports | DONE | Per-inspection report; distributions and per-product breakdown in the summary |
| Analytics dashboard | Inspection statistics | DONE | Summary and `GET /inspections/analytics/by-category` |
| Analytics dashboard | Operational insights | DONE | Deterministic current-versus-previous-window insights |

### Performance metrics

| Metric | Status | Where it is computed or measured, and the value |
|---|---|---|
| Detection accuracy | DONE | Final-test reports; mean over 15 categories 91.5% |
| Precision | DONE | Final-test reports; mean 97.7% |
| Recall | DONE | Mean 91.1%; per category in Milestone 4 |
| F1-score | DONE | Mean 93.8% |
| mAP score | DONE, labelled | Image-level AP (area under the image-level precision-recall curve): mean 0.9946. Box AP@0.5 of the anomaly-map boxes: mean 0.200. Neither is a trained detector's mAP. |
| Inspection automation rate | DONE | `automation_rate` in `GET /inspections/analytics/summary`: share of decided inspections decided PASS or FAIL. A live value; none is stored in the repository. |
| Defect identification accuracy | DONE | Reported as recall (the reports state `defect_identification_accuracy` equals recall) |
| False defect detection rate | DONE | Reported as FPR; mean 7.7% |
| Image processing speed | DONE | Final-test reports: 51 ms (toothbrush) to 1943 ms (grid) per image on the development machine |
| Inspection response time | DONE | `processing_time_ms` and `ai_inference_time_ms` per inspection, shown on the dashboard; on the AWS VM Grid takes 3.9 s warm ([DEPLOY.md](DEPLOY.md)) |
| Concurrent inspection capability | PARTIAL | One synchronous worker; DEPLOY.md expects roughly 2 inspections per second on 8 cores for the lighter models. `backend/scripts/concurrency_check.py` exists, but no result of it is stored. |
"""

STACK = """\
## Appendix B: Technology stack as built

Checked against `backend/requirements.txt`, `frontend/package.json`, the Dockerfiles and `docs/DEPLOY.md`.

| Planned | As built |
|---|---|
| FastAPI | Used (backend API) |
| React.js or Next.js | React 19 with Vite; Next.js not used |
| PostgreSQL and MongoDB | PostgreSQL (18 in Docker); MongoDB not used |
| OpenCV | Used (decoding, resizing, connected components, enhancement preview) |
| TensorFlow | Not used |
| PyTorch | Used, CPU build |
| YOLO | Not used; boxes are derived from the anomaly map |
| CNN models | Frozen ImageNet ResNet-18 and Wide ResNet-50-2 as feature extractors |
| scikit-learn | Used (AUROC and average precision in evaluation) |
| NumPy | Used |
| Pandas | Not used |
| Image augmentation pipeline | Not used; synthetic defects on held-out normal images are used for model selection only |
| Defect segmentation models | Not used; anomaly maps give heatmaps and boxes |
| Docker | Docker Compose with web, backend and db services |
| AWS or Azure | AWS EC2, ap-south-1 (Mumbai) |
| Tailwind CSS | Not used; CSS Modules |
| JWT | Used (PyJWT, HS256) |
| Git and GitHub | Used |
| Postman | Collection and environment in `postman/` |
| Logging and monitoring (optional) | Python standard logging; no monitoring service |
"""

OPERATIONS = """\
## Setup

Models, dataset, Docker Compose and local development setup are described step by step in
[README.md](../README.md#setup).

## Testing

The backend suite (`python -m pytest` in `backend/`) and the frontend suite (`npx vitest run` in `frontend/`), the
test database they use and the Postman collection are described in [README.md](../README.md#testing).

## Known limitations

The current limitations of models, localization, severity, confidence, data and deployment are listed in
[README.md](../README.md#known-limitations).

## Deployment

Deployment on a Linux VM with Docker Compose, the tested configurations (including AWS), backup and the redeploy
procedure are in [DEPLOY.md](DEPLOY.md).

## Licence

The code is licensed under the MIT License ([LICENSE.txt](../LICENSE.txt)). The dataset, the pretrained weights and
the trained model files are not covered by it; see [README.md](../README.md#licence).
"""


def main() -> None:
    for relative in LINKED:
        if not (REPO / relative).is_file():
            fail(f"missing linked file: {relative}")

    chapters = [embed(path) for path in MILESTONES]
    shared = [embed(path) for path in SHARED]

    titles = ["Introduction"] + [t for t, _ in chapters] + [
        "Appendix A: Requirements traceability", "Appendix B: Technology stack as built"] + [t for t, _ in shared] + [
        "Setup", "Testing", "Known limitations", "Deployment", "Licence"]
    contents = "\n".join(f"{i}. [{title}](#{slug(title)})" for i, title in enumerate(titles, 1))

    parts = [
        "# VisionInspect AI: Project documentation",
        "This file is generated by `docs/build_full_doc.py` from the milestone files in `docs/milestones/` and the "
        "shared documents in `docs/`. Edit those files, not this one.",
        "## Contents\n\n" + contents,
        INTRODUCTION.strip("\n"),
        *[body for _, body in chapters],
        TRACEABILITY.strip("\n"),
        STACK.strip("\n"),
        *[body for _, body in shared],
        OPERATIONS.strip("\n"),
    ]
    text = "\n\n".join(part.strip("\n") for part in parts) + "\n"
    (REPO / OUTPUT).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
