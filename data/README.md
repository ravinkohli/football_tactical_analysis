# Data

All datasets here are tracked with [DVC](https://dvc.org/), not git. The actual
bytes live in the DVC remote (`localremote` → `/tmp/dvc-store`); git only stores
the small `.dvc` / `dvc.lock` pointers. To materialise the files locally:

```bash
dvc pull                 # fetch everything (raw + converted)
dvc pull data/issia_coco # or just one target
```

## Layout

```
data/
├── issia_raw/                      # raw ISSIA-CNR dataset            (DVC-tracked)
├── issia_coco/                     # ISSIA converted to COCO          (pipeline output)
├── SoccerPlayerDetection_raw/      # raw Soccer Player Detection set  (DVC-tracked)
└── SoccerPlayerDetection_coco/     # SPD converted to COCO            (pipeline output)
```

The `*_raw` folders are imported data (`dvc add`). The `*_coco` folders are
**generated** by the DVC pipeline (`dvc.yaml`) from the raw data — never edit
them by hand; change the converter or `params.yaml` and re-run `dvc repro`.

---

## `issia_raw` — ISSIA-CNR Soccer Dataset

Six synchronised broadcast cameras of a single match.

| | |
|---|---|
| Videos | `filmrole1.avi` … `filmrole6.avi` (DivX 5, 1920×1080, 25 fps) |
| Annotations | `Annotation Files/*.xgtf` (ViPER XML), `ID-{k}` pairs with `filmrole{k}` |
| Labels | per-frame `Person` bounding boxes + a per-frame `BALL` point |

> ⚠️ OpenCV's bundled build cannot decode DivX — the converter uses `ffmpeg`.
> The annotation frame numbering runs ~25 frames past the last decodable frame;
> those frames are dropped during conversion.

## `SoccerPlayerDetection_raw` — Soccer Player Detection (Lu et al., BMVC 2017)

Two broadcast games, already extracted to JPG frames.

| | |
|---|---|
| Images | `DataSet_001/` (game 1, 1495 frames), `DataSet_002/` (game 2, 524 frames), 1280×720, 30 fps |
| Annotations | `annotation_1.mat`, `annotation_2.mat` — MATLAB struct array `annot` with `ImgName` + `BBox` |
| Labels | player bounding boxes only, in `[x1, y1, x2, y2]` corner format |
| Extras | `ReadDemo.m` (MATLAB viewer), `readme.txt` (original notes) |

Frame filenames are the 0-based frame number (`0186.jpg` = frame 186) and
collide across the two games.

---

## Converted COCO datasets

Both converters emit the standard COCO detection layout:

```
<dataset>_coco/
├── annotations.json    # COCO: info / categories / images / annotations
└── images/             # one JPG per image, file_name matches annotations.json
```

| Output | Images | Annotations | Categories | Per-image extras |
|---|---|---|---|---|
| `issia_coco` | 3166 | 26 834 | `person`, `ball` | `camera_id`, `frame_index`, `video` |
| `SoccerPlayerDetection_coco` | 2019 | 22 586 | `person` | `game_id`, `frame_index` |

Bounding boxes are COCO `[x, y, w, h]` (top-left + size), clipped to the image.

### Quick load

```python
from pycocotools.coco import COCO          # or: import json; json.load(open(...))
coco = COCO("data/issia_coco/annotations.json")
img = coco.loadImgs(coco.getImgIds()[0])[0]
anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))
# image file: data/issia_coco/images/<img["file_name"]>
```

See [`src/data/README.md`](../src/data/README.md) for how the conversions are
run and configured.
