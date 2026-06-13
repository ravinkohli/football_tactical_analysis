"""Convert the Soccer Player Detection dataset (Lu et al., BMVC 2017) to COCO.

The raw dataset ships two games, each a folder of already-extracted JPG frames
plus a MATLAB annotation file:

* ``DataSet_001/`` <- ``annotation_1.mat`` (game 1, 1495 frames)
* ``DataSet_002/`` <- ``annotation_2.mat`` (game 2, 524 frames)

Each ``.mat`` holds a struct array ``annot`` with fields ``ImgName`` (e.g.
``"0186.jpg"``, the 0-based frame number at 30fps) and ``BBox`` (an Nx4 matrix
of player boxes in ``[x1, y1, x2, y2]`` corner format). There is a single object
class: ``person``.

Frame filenames collide across the two games (both contain e.g. ``0216.jpg``),
so output images are prefixed with the game id (``game1_0186.jpg``). Images are
copied as-is (no decoding needed); we emit a single COCO JSON with one
``person`` category.

Usage:
    python -m src.data.convert_soccer_player_detection \
        --raw-dir data/SoccerPlayerDetection_raw \
        --out-dir data/SoccerPlayerDetection_coco
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import scipy.io as sio

CATEGORIES = [{"id": 1, "name": "person", "supercategory": "soccer"}]
PERSON_CAT_ID = 1

# game id -> (annotation .mat, image folder)
GAMES = {
    1: ("annotation_1.mat", "DataSet_001"),
    2: ("annotation_2.mat", "DataSet_002"),
}


def frame_number(img_name: str) -> int:
    """Return the integer frame number from a name like ``"0186.jpg"``."""
    return int(Path(img_name).stem)


def load_entries(
    mat_path: Path,
) -> list[tuple[str, list[tuple[float, float, float, float]]]]:
    """Load ``(ImgName, [(x1, y1, x2, y2), ...])`` tuples from an annotation .mat.

    ``annot`` is a ``(1, N)`` MATLAB struct array; each cell's ``BBox`` is an
    ``M x 4`` array of corner-format boxes.
    """
    annot = sio.loadmat(mat_path)["annot"][0]
    entries = []
    for cell in annot:
        # ImgName comes back as a 1-element array of str.
        img_name = str(cell["ImgName"][0])
        boxes = [tuple(float(v) for v in row) for row in cell["BBox"]]
        entries.append((img_name, boxes))
    return entries


def clip_corners(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Clamp a corner box to the image, returning a COCO ``(x, y, w, h)`` box."""
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    return x1, y1, x2 - x1, y2 - y1


def convert(raw_dir: Path, out_dir: Path, sample_rate: int) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict] = []
    coco_annots: list[dict] = []
    img_id = 0
    ann_id = 0

    for game, (mat_name, ds_name) in sorted(GAMES.items()):
        src_dir = raw_dir / ds_name
        entries = load_entries(raw_dir / mat_name)
        entries.sort(key=lambda e: frame_number(e[0]))
        sampled = entries[::sample_rate]
        print(f"game {game}: {len(entries)} annotated frames -> {len(sampled)} sampled")

        missing = 0
        for img_name, boxes in sampled:
            src = src_dir / img_name
            if not src.exists():
                missing += 1
                continue
            frame = frame_number(img_name)
            file_name = f"game{game}_{img_name}"
            image = cv2.imread(str(src))
            height, width = image.shape[:2]
            shutil.copyfile(str(src), str(images_dir / file_name))

            img_id += 1
            coco_images.append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "game_id": game,
                    "frame_index": frame,
                }
            )
            for bx1, by1, bx2, by2 in boxes:
                x, y, w, h = clip_corners(bx1, by1, bx2, by2, width, height)
                if w <= 0 or h <= 0:
                    continue
                ann_id += 1
                coco_annots.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": PERSON_CAT_ID,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
        if missing:
            print(f"  ! {missing} annotated frames had no image file and were skipped")

    coco = {
        "info": {
            "description": "Soccer Player Detection dataset (Lu et al. 2017) in COCO",
            "source": str(raw_dir),
            "sample_rate": sample_rate,
        },
        "categories": CATEGORIES,
        "images": coco_images,
        "annotations": coco_annots,
    }
    ann_path = out_dir / "annotations.json"
    with ann_path.open("w") as f:
        json.dump(coco, f)
    print(
        f"\nWrote {len(coco_images)} images and {len(coco_annots)} annotations "
        f"to {ann_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-dir", type=Path, default=Path("data/SoccerPlayerDetection_raw")
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/SoccerPlayerDetection_coco")
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="keep every Nth annotated frame (default: 1, keep all)",
    )
    args = p.parse_args()
    convert(raw_dir=args.raw_dir, out_dir=args.out_dir, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
