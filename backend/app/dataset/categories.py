"""The 15 MVTec AD categories, exactly as named by the dataset's top-level folders.

The single place this list is written down. app.dataset.service still discovers categories
by scanning DATASET_ROOT (so the dataset browser reflects what is actually on disk); this
constant is for validating values that are stored without a dataset lookup, such as a
product's MVTec category. Which categories have a served AI model is a separate question,
answered by app.ai.inference.serving.get_supported_categories().
"""

from typing import Literal, get_args

MvtecCategory = Literal[
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

MVTEC_CATEGORIES: tuple[str, ...] = get_args(MvtecCategory)
