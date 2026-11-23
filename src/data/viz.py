"""Visualisation helpers for the converted COCO datasets.

Designed for quick inspection from a notebook (see
``notebooks/explore_dataset.ipynb``). Works with the output of either converter
in this package — anything laid out as ``<dir>/annotations.json`` + ``<dir>/images/``.

Example
-------
>>> from src.data.viz import CocoView
>>> view = CocoView("data/issia_coco")
>>> view.show(0)                 # first image with its boxes
>>> view.show("cam1_frame000364.jpg")
>>> view.show(random=True)       # a random frame
>>> len(view), view.category_names
"""

from __future__ import annotations

import json
import random as _random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Stable per-category colours (RGB 0-1), enough for the small label sets here.
_PALETTE = [
    (0.20, 0.80, 0.20),  # green
    (0.90, 0.20, 0.20),  # red
    (0.20, 0.50, 0.95),  # blue
    (0.95, 0.75, 0.10),  # amber
    (0.70, 0.30, 0.85),  # purple
    (0.10, 0.75, 0.75),  # teal
]


class CocoView:
    """A loaded COCO dataset with helpers to render a frame and its boxes."""

    def __init__(self, coco_dir: str | Path):
        self.root = Path(coco_dir)
        self.images_dir = self.root / "images"
        with (self.root / "annotations.json").open() as f:
            self.coco = json.load(f)

        self.images: list[dict] = self.coco["images"]
        self.categories: list[dict] = self.coco["categories"]
        self.cat_name = {c["id"]: c["name"] for c in self.categories}
        self.cat_color = {
            c["id"]: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(self.categories)
        }

        self._anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in self.coco["annotations"]:
            self._anns_by_image[ann["image_id"]].append(ann)
        self._by_file = {img["file_name"]: img for img in self.images}

    def __len__(self) -> int:
        return len(self.images)

    @property
    def category_names(self) -> list[str]:
        return [c["name"] for c in self.categories]

    def annotations(self, image_id: int) -> list[dict]:
        """All annotations for a given image id."""
        return self._anns_by_image[image_id]

    def _resolve(self, ref: int | str | None, random: bool) -> dict:
        """Resolve an image by list index (int), file name (str), or random."""
        if random:
            return _random.choice(self.images)
        if ref is None:
            return self.images[0]
        if isinstance(ref, str):
            if ref not in self._by_file:
                raise KeyError(f"No image named {ref!r} in {self.root}")
            return self._by_file[ref]
        return self.images[ref]  # treat int as index into the image list

    def show(
        self,
        ref: int | str | None = None,
        *,
        random: bool = False,
        ax: Axes | None = None,
        show_labels: bool = True,
        linewidth: float = 1.5,
        figsize: tuple[float, float] = (14, 8),
    ) -> Axes:
        """Draw an image with its bounding boxes and return the matplotlib axes.

        ``ref`` is a 0-based index into the image list, or a ``file_name``.
        Pass ``random=True`` to pick a random frame instead.
        """
        img = self._resolve(ref, random)
        path = self.images_dir / img["file_name"]
        array = cv2.imread(str(path))
        if array is None:
            raise FileNotFoundError(f"Could not read image {path}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.imshow(array)

        anns = self.annotations(img["id"])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            color = self.cat_color.get(ann["category_id"], (1.0, 1.0, 1.0))
            ax.add_patch(
                plt.Rectangle(
                    (x, y), w, h, fill=False, edgecolor=color, linewidth=linewidth
                )
            )
            if show_labels:
                ax.text(
                    x,
                    y - 2,
                    self.cat_name.get(ann["category_id"], "?"),
                    color="black",
                    fontsize=7,
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 0.5},
                )

        meta = ", ".join(
            f"{k}={img[k]}" for k in ("camera_id", "game_id", "frame_index") if k in img
        )
        ax.set_title(f"{img['file_name']}  ({len(anns)} boxes)  {meta}".strip())
        ax.axis("off")
        return ax


def show_frame(coco_dir: str | Path, ref: int | str | None = None, **kwargs) -> Axes:
    """One-shot convenience wrapper: ``show_frame("data/issia_coco", 0)``."""
    return CocoView(coco_dir).show(ref, **kwargs)
