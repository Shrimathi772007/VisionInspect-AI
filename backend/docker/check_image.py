"""Build-time checks for the backend image (run once by the Dockerfile; the build fails if any check fails).

1. Every `name==version` pin in requirements.txt is installed at exactly that version (a local label such as
   "+cpu" is ignored when comparing).
2. torch is the CPU-only build (no CUDA runtime).
3. The two Grid study modules that serving imports from scripts/experiments/patchcore_wrn50/ have the SHA-256
   values pinned in app/ai/inference/serving.py, i.e. they were copied into the image byte-identical.

Reads files only; imports nothing from the app.
"""

import hashlib
import importlib.metadata
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
STUDY_DIR = BACKEND_DIR / "scripts" / "experiments" / "patchcore_wrn50"
SERVING = BACKEND_DIR / "app" / "ai" / "inference" / "serving.py"
STUDY_FILES = {
    "wrn50_study_logic.py": "WRN50_STUDY_LOGIC_SHA256",
    "grid_full320_logic.py": "GRID_FULL320_LOGIC_SHA256",
}


def check_pins() -> list[str]:
    errors = []
    pins = 0
    for line in (BACKEND_DIR / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        name = re.sub(r"\[.*\]", "", name).strip()
        installed = importlib.metadata.version(name)
        pins += 1
        if installed.split("+", 1)[0] != version.strip():
            errors.append(f"{name}: requirements.txt pins {version.strip()}, installed {installed}")
    print(f"requirements.txt: {pins} pins checked")
    return errors


def check_torch() -> list[str]:
    import torch

    print(f"torch {torch.__version__} (CUDA runtime: {torch.version.cuda})")
    if torch.version.cuda is not None or not torch.__version__.endswith("+cpu"):
        return [f"torch {torch.__version__} is not the CPU-only build"]
    return []


def check_study_files() -> list[str]:
    errors = []
    serving = SERVING.read_text()
    for filename, constant in STUDY_FILES.items():
        match = re.search(rf'^{constant} = "([0-9a-f]{{64}})"', serving, re.MULTILINE)
        if match is None:
            errors.append(f"{constant} not found in {SERVING.name}")
            continue
        actual = hashlib.sha256((STUDY_DIR / filename).read_bytes()).hexdigest()
        print(f"{filename}: {actual}")
        if actual != match.group(1):
            errors.append(f"{filename}: expected SHA-256 {match.group(1)}, found {actual}")
    return errors


def main() -> int:
    errors = check_pins() + check_torch() + check_study_files()
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not errors:
        print("Image checks passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
