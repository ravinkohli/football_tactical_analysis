# `src/data` — dataset converters

Scripts that turn the raw football datasets in [`data/`](../../data/) into a
standard **COCO detection** format (`annotations.json` + `images/`). They are
wired into the DVC pipeline ([`dvc.yaml`](../../dvc.yaml)), so the normal way to
run them is `dvc repro`; they can also be run directly for one-off experiments.

| Module | Raw input | Output | Categories |
|---|---|---|---|
| [`convert_issia.py`](convert_issia.py) | `data/issia_raw` (DivX `.avi` + ViPER `.xgtf`) | `data/issia_coco` | `person`, `ball` |
| [`convert_soccer_player_detection.py`](convert_soccer_player_detection.py) | `data/SoccerPlayerDetection_raw` (`.jpg` + `.mat`) | `data/SoccerPlayerDetection_coco` | `person` |

## Running via DVC (recommended)

The pipeline reads its knobs from [`params.yaml`](../../params.yaml) and only
re-runs a stage when its code, inputs, or params change.

```bash
dvc repro                              # build any out-of-date stage
dvc repro convert_issia                # build just one stage
dvc push                               # upload outputs to the DVC remote
```

To change behaviour, edit `params.yaml` (then `dvc repro`):

```yaml
convert_issia:
  sample_rate: 5     # keep every Nth annotated frame
  ball_size: 20      # px box drawn around each ball point
convert_soccer_player_detection:
  sample_rate: 1     # 1 = keep every provided frame
```

## Running directly

Each script is an `argparse` CLI / module:

```bash
python -m src.data.convert_issia \
    --raw-dir data/issia_raw --out-dir data/issia_coco \
    --sample-rate 5 --ball-size 20 [--no-images]

python -m src.data.convert_soccer_player_detection \
    --raw-dir data/SoccerPlayerDetection_raw \
    --out-dir data/SoccerPlayerDetection_coco --sample-rate 1
```

`convert_issia --no-images` writes `annotations.json` only, skipping the (slow)
ffmpeg frame extraction.

## How they work

**`convert_issia`** — treats each `filmrole{k}.avi` as camera `k`, paired with
its `ID-{k}` `.xgtf`. It parses per-frame `Person` boxes and `BALL` points,
keeps every Nth annotated frame, and extracts exactly those frames with
**ffmpeg** (OpenCV can't decode DivX). Ball points become a small fixed-size
box. Frames whose index exceeds the video length are dropped.

**`convert_soccer_player_detection`** — loads the `annot` struct array from each
`.mat` with `scipy.io.loadmat`, converts `BBox` corner format
`[x1,y1,x2,y2]` → COCO `[x,y,w,h]`, and copies the existing JPGs (prefixing the
game id, since frame names collide across games).

Both clip every box to the image bounds and drop degenerate (zero-area) boxes.

## Requirements

- Python deps are declared in [`pyproject.toml`](../../pyproject.toml)
  (`opencv-python`, `scipy`, …); run `uv sync`.
- **`ffmpeg`** must be on `PATH` for `convert_issia` (e.g. `brew install ffmpeg`).
