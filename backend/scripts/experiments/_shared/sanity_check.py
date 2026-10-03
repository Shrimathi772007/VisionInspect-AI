"""Ad hoc Phase 6 sanity check - NOT part of the test suite, not committed.

Runs predict_image on one real known-good and one real known-defective MVTec
bottle test image, using the existing trained model and Phase 5 threshold
logic exactly as-is. Ground truth (folder name) is only used to LABEL the
printed report - it is never passed into predict_image or used to influence
the prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
BACKEND_DIR = Path(r"c:\Users\SHRIMATHI S\Documents\VisionInspect-AI\backend")
sys.path.insert(0, str(BACKEND_DIR))

from app.ai.inference import predict_image  # noqa: E402

DATASET_ROOT = BACKEND_DIR / ".." / "dataset"

good_image = sorted((DATASET_ROOT / "bottle" / "test" / "good").glob("*.png"))[0]
defective_image = sorted((DATASET_ROOT / "bottle" / "test" / "broken_large").glob("*.png"))[0]

for label, image_path in [("known good (ground truth, not used by AI)", good_image),
                           ("known defective (ground truth, not used by AI)", defective_image)]:
    result = predict_image(image_path, "bottle")
    print(f"Image: {image_path.name}  [{label}]")
    print(f"  category:            {result.category}")
    print(f"  reconstruction_error: {result.reconstruction_error:.6f}")
    print(f"  threshold:            {result.threshold:.6f}")
    print(f"  AI prediction:        {result.prediction}")
    print()
