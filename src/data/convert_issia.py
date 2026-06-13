"""Convert the ISSIA-CNR soccer dataset to COCO detection format.

The raw dataset ships 6 videos (``filmrole1.avi`` .. ``filmrole6.avi``, DivX 5,
1920x1080, 25fps) and 6 matching ViPER ground-truth files in ``Annotation
Files/`` (``... ID-1 ...`` .. ``... ID-6 ...``). Each video is treated as a
separate *camera*. The annotation files provide, per frame:

* ``Person`` objects with a ``LOCATION`` bounding box (x, y, width, height).
* a ``BALL`` object with ``BallPos`` points (a single x/y per frame).

This script collects the annotated frames for each camera, keeps every Nth one
(``--sample-rate``), extracts exactly those frames from the ``.avi`` with ffmpeg
(OpenCV cannot decode DivX in the bundled build), and writes a single COCO JSON
with two categories: ``person`` and ``ball``. Ball points are converted to a
small fixed-size box centred on the point.

Usage:
    python -m src.data.convert_issia \
        --raw-dir data/issia_raw --out-dir data/issia_coco --sample-rate 5
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ViPER XML namespaces used in the .xgtf files.
NS = {
    "v": "http://lamp.cfar.umd.edu/viper#",
    "d": "http://lamp.cfar.umd.edu/viperdata#",
}
OBJECT = f"{{{NS['v']}}}object"
BBOX = f"{{{NS['d']}}}bbox"
POINT = f"{{{NS['d']}}}point"

CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "issia"},
    {"id": 2, "name": "ball", "supercategory": "issia"},
]
PERSON_CAT_ID = 1
BALL_CAT_ID = 2


@dataclass
class FrameAnnots:
    """Annotations gathered for a single (camera, frame) pair."""

    persons: list[tuple[float, float, float, float]] = field(default_factory=list)
    ball: tuple[float, float] | None = None  # (x, y) point


def find_cameras(raw_dir: Path) -> dict[int, tuple[Path, Path]]:
    """Map camera id -> (video path, annotation path).

    Cameras are 1-indexed: ``filmrole{k}.avi`` pairs with the ``... ID-{k} ...``
    annotation file.
    """
    ann_dir = raw_dir / "Annotation Files"
    cameras: dict[int, tuple[Path, Path]] = {}
    for video in sorted(raw_dir.glob("filmrole*.avi")):
        m = re.search(r"filmrole(\d+)", video.name)
        if not m:
            continue
        cam = int(m.group(1))
        matches = list(ann_dir.glob(f"*ID-{cam} *.xgtf"))
        if not matches:
            raise FileNotFoundError(f"No annotation file for camera {cam} (ID-{cam})")
        cameras[cam] = (video, matches[0])
    if not cameras:
        raise FileNotFoundError(f"No filmrole*.avi videos found in {raw_dir}")
    return cameras


def clip_bbox(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Clamp an (x, y, w, h) box to the image, returning a non-negative box."""
    x0 = min(max(x, 0.0), width)
    y0 = min(max(y, 0.0), height)
    x1 = min(max(x + w, 0.0), width)
    y1 = min(max(y + h, 0.0), height)
    return x0, y0, x1 - x0, y1 - y0


def first_frame(framespan: str) -> int:
    """Return the starting frame of a ViPER ``framespan`` like ``"364:366"``."""
    return int(framespan.split(":")[0])


def parse_annotations(xgtf_path: Path) -> dict[int, FrameAnnots]:
    """Parse a .xgtf file into a ``{frame_index: FrameAnnots}`` mapping.

    Each ``data:bbox`` / ``data:point`` carries its own ``framespan`` (one entry
    per frame), so we key strictly on the frame the entry belongs to.
    """
    root = ET.parse(xgtf_path).getroot()
    frames: dict[int, FrameAnnots] = defaultdict(FrameAnnots)
    for obj in root.iter(OBJECT):
        name = (obj.get("name") or "").upper()
        if name == "PERSON":
            for bb in obj.iter(BBOX):
                frame = first_frame(bb.get("framespan"))
                frames[frame].persons.append(
                    (
                        float(bb.get("x")),
                        float(bb.get("y")),
                        float(bb.get("width")),
                        float(bb.get("height")),
                    )
                )
        elif name == "BALL":
            for pt in obj.iter(POINT):
                frame = first_frame(pt.get("framespan"))
                frames[frame].ball = (float(pt.get("x")), float(pt.get("y")))
    return frames


def probe_resolution(video: Path) -> tuple[int, int]:
    """Return (width, height) of a video via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def extract_frames(
    video: Path, frame_indices: list[int], dest: dict[int, Path]
) -> set[int]:
    """Extract the requested (0-based) frames from ``video``.

    Decodes the whole video once into a temp dir; ffmpeg numbers the output
    sequentially from 1, so decode-order index ``n`` (0-based) lands in
    ``{n + 1:06d}.jpg``. Frames whose index exceeds the video's actual length
    (ISSIA annotations run a little past the last decodable frame) are simply
    skipped. Returns the set of frame indices successfully written.
    """
    if not frame_indices:
        return set()
    written: set[int] = set()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vsync",
                "0",
                "-pix_fmt",
                "yuvj420p",
                "-q:v",
                "2",
                str(tmp_dir / "%06d.jpg"),
            ],
            check=True,
        )
        for n in frame_indices:
            src = tmp_dir / f"{n + 1:06d}.jpg"
            if src.exists():
                shutil.move(str(src), str(dest[n]))
                written.add(n)
    return written


def convert(
    raw_dir: Path,
    out_dir: Path,
    sample_rate: int,
    ball_size: float,
    no_images: bool,
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cameras = find_cameras(raw_dir)
    coco_images: list[dict] = []
    coco_annots: list[dict] = []
    img_id = 0
    ann_id = 0

    for cam, (video, xgtf) in sorted(cameras.items()):
        width, height = probe_resolution(video)
        frames = parse_annotations(xgtf)
        annotated = sorted(frames)
        sampled = annotated[::sample_rate]
        print(
            f"camera {cam}: {len(annotated)} annotated frames "
            f"-> {len(sampled)} sampled (every {sample_rate})"
        )

        names = {frame: f"cam{cam}_frame{frame:06d}.jpg" for frame in sampled}
        if no_images:
            available = sampled
        else:
            dest = {frame: images_dir / names[frame] for frame in sampled}
            written = extract_frames(video, sampled, dest)
            available = [frame for frame in sampled if frame in written]
            if len(available) != len(sampled):
                print(
                    f"  ! {len(sampled) - len(available)} sampled frames exceed "
                    f"the video length and were dropped"
                )

        for frame in available:
            img_id += 1
            coco_images.append(
                {
                    "id": img_id,
                    "file_name": names[frame],
                    "width": width,
                    "height": height,
                    "camera_id": cam,
                    "frame_index": frame,
                    "video": video.name,
                }
            )
            fa = frames[frame]
            for px, py, pw, ph in fa.persons:
                x, y, w, h = clip_bbox(px, py, pw, ph, width, height)
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
            if fa.ball is not None:
                bx, by = fa.ball
                x, y, w, h = clip_bbox(
                    bx - ball_size / 2,
                    by - ball_size / 2,
                    ball_size,
                    ball_size,
                    width,
                    height,
                )
                if w > 0 and h > 0:
                    ann_id += 1
                    coco_annots.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": BALL_CAT_ID,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )

    coco = {
        "info": {
            "description": "ISSIA-CNR soccer dataset converted to COCO",
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
    p.add_argument("--raw-dir", type=Path, default=Path("data/issia_raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/issia_coco"))
    p.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="keep every Nth annotated frame (default: 5)",
    )
    p.add_argument(
        "--ball-size",
        type=float,
        default=20.0,
        help="side length (px) of the box drawn around each ball point",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        help="write COCO JSON only, skip ffmpeg frame extraction",
    )
    args = p.parse_args()
    convert(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        sample_rate=args.sample_rate,
        ball_size=args.ball_size,
        no_images=args.no_images,
    )


if __name__ == "__main__":
    main()
