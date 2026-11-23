# Football Visual Analytics — Project Plan

**Product goal:** From soccer match video, compute physical and technical stats (run speed, distance, sprints, possession, pass accuracy) and produce a per-player performance diagnosis — strengths and weaknesses.

**Meta-goal (the real reason for the project):** Learn the modern ML lifecycle hands-on — distributed training, finetuning, experiment tracking, and model monitoring — by building something realistic. The football CV is the *vehicle*; the lifecycle skills are the destination. Be ambitious about the pipeline, modest about model accuracy.

---

## 1. Decisions log

These are settled from our scoping; everything below builds on them.

| Area | Decision | Why |
|---|---|---|
| Sport | Soccer (association football) | Best open datasets (ISSIA, SoccerNet, StatsBomb) |
| Hardware | Apple M5 Pro (MPS) for dev; rented cloud CUDA for heavy/distributed runs | Mac can't do CUDA/multi-GPU; local↔cloud split is the realistic workflow |
| Primary dataset | ISSIA-CNR (6 fixed cameras, ~18k frames, 1080p@25fps, player/ball/referee boxes) | Tracking + calibration friendly; fixed cameras simplify homography |
| Later data | SoccerNet (action spotting, calibration), StatsBomb open data (events) | Needed for passes/events and diagnosis benchmarks |
| Week-1 model | RT-DETRv2 via Ultralytics, `device=mps` | Transformer (rich for learning) but runs on MPS via portable deformable attention; same API as YOLO |
| Fallback model | YOLOv8n | Instant baseline if RT-DETR crawls on MPS |
| Serious-path model | HuggingFace RT-DETRv2 + Accelerate + PEFT | Exposes the training loop → real distributed/finetuning lessons; COCO-native |
| Optional comparison | RF-DETR (cloud only) | CUDA-only deformable attention kernel — won't run well on MPS |
| Annotation format | **COCO JSON canonical**, derive YOLO view | COCO is richer, HF-native, gives AP_small (the ball); YOLO derived for Ultralytics |
| Tracking | ByteTrack | Bundled with Ultralytics; strong baseline |
| Team assignment | Color-histogram baseline → SigLIP-embedding clustering | Embeddings robust to shadow/kit similarity |
| Homography | Manual per-camera calibration (once) | ISSIA cameras are fixed → calibrate once, reuse |
| Tracking server | MLflow (experiments + model registry) | Self-hosted = production feel; owns all weights |
| Data versioning | DVC with pipeline stages | Ties raw → processed → derived with pinned hashes |
| Architecture | Config-driven, model-/device-agnostic | Swap model/device/backend via YAML, compare as tracked experiments |

---

## 2. System architecture — the dependency chain

No model outputs a "stat." Models do **perception**, geometry produces **stats**, analytics produces **diagnosis**. Each stat depends on different links, so order matters:

```
   detection + tracking  ──►  player trajectories (pixels)
            + homography  ──►  trajectories in METERS      ◄── the linchpin
       + team assignment  ──►  who is on which team
   + ball track + events  ──►  possession, passes
                          ──►  metrics: speed, distance, pass %
                          ──►  diagnosis: strengths / weaknesses
```

**Homography is the gate.** You cannot compute speed in pixels (near players move more pixels than far ones for the same real distance). Until pixel→meter mapping works, no physical stat is valid. ISSIA's fixed cameras let you calibrate once per camera (~6 pitch-line correspondences into OpenCV `findHomography`) and skip the hard per-frame case.

---

## 3. Repository structure

```
football-analytics/
  configs/                 # all hyperparams/choices — nothing hardcoded
    week1_rtdetr.yaml
  data/                    # DVC-tracked, gitignored content
    issia_raw/             # raw download (DVC input)
    issia_coco/            # canonical processed (COCO) — DVC output
    issia_yolo/            # derived YOLO view — DVC output
  src/
    convert_issia.py       # raw → COCO JSON
    coco_to_yolo.py        # COCO → YOLO (for Ultralytics backend)
    models/                # WRAPPER code, not architectures
      base.py              # interface: train() / predict() / load()
      ultralytics_backend.py
      hf_backend.py        # added in the HF phase
    calibrate.py           # per-camera homography
    track.py               # detection + tracking → trajectories
    teams.py               # team assignment
    stats.py               # trajectories(+teams) → metrics
    diagnose.py            # metrics → performance profile
    train.py
    infer.py
  dvc.yaml                 # pipeline stages (prepare, etc.)
  notebooks/
```

**No top-level `models/` for architectures** — wrapped models own those. Trained weights live in the **MLflow registry**, not git. Add a gitignored `models/production/` staging dir only when you reach serving.

---

## 4. MLOps foundations (set up before any serious training)

- **Experiment tracking:** MLflow server (`mlflow server --host 127.0.0.1 --port 5000`). Log params, metrics, artifacts, and register promising models with stages (Staging → Production). Use **manual** logging or Ultralytics autolog — never both.
- **Data versioning:** DVC with a pipeline. Raw is a tracked *input*; the converter is a tracked *dependency*; processed COCO/YOLO are tracked *outputs*. Commit `dvc.lock`. Reproducibility chain: **git commit ↔ dvc.lock ↔ MLflow run**. For multi-GB raw, consider `dvc import-url` to pin source + checksum without storing it yourself.
- **Config-driven:** every choice (model, device, epochs, dataset version) in YAML.
- **Pluggable pipeline (important nuance):** Ultralytics owns its training loop; HF hands you the loop. So you can't put them behind one *training* interface. Instead share the **data layer (DVC), tracking (MLflow), and eval (COCO mAP / torchmetrics)**, and have **two trainer backends** — `ultralytics` (quick baseline) and `hf` (the real loop) — selected by config.

---

## 5. Product milestone roadmap

| # | Milestone | Needs | Deliverable | Data | Validation |
|---|---|---|---|---|---|
| M1 | Detection + tracking | detector + ByteTrack | Player trajectories (pixels) with IDs | ISSIA | ISSIA box GT (mAP, MOTA) |
| M2 | Homography | per-camera calibration | Trajectories in meters; top-down "radar" view | ISSIA | ISSIA position GT |
| M3 | Physical stats + profile | M2 + trajectory smoothing | Speed, distance, sprints, heatmaps; physical performance profile | ISSIA | Speed vs GT positions |
| M4 | Team assignment | crops → color/embedding clustering, per-track vote | Each track labeled to a team | ISSIA (refs labeled) | Visual overlay (no GT labels) |
| M5 | Ball + possession + passes | ball tracking + possession logic + event detection | Pass count, completion %, possession % | SoccerNet / StatsBomb | StatsBomb events |
| M6 | Full diagnosis | M3–M5 + benchmarks | Per-player strengths/weaknesses report | SoccerNet + StatsBomb | Benchmark comparison |

**Sequencing logic:** physical stats (M1–M3) are reliable and fully supported by ISSIA — start there. Team assignment (M4) is mandatory before anything team-relative. Passes and true diagnosis (M5–M6) need labeled event data ISSIA doesn't have.

**Key gotchas per milestone:**
- M2: pitch-line correspondences must be accurate — small calibration error → wrong meters everywhere.
- M3: **smooth trajectories (Kalman / Savitzky-Golay) before differentiating** — raw Δpos/Δt gives 80 km/h players.
- M4: it's not 2 clusters — Team A, Team B, 2 goalkeepers, referees. Exclude refs (ISSIA labels them), treat GKs as outliers; vote per track to stop flicker.
- M5: ball is tiny/fast/occluded — your hardest perception target.

---

## 6. MLOps pillars woven across phases

- **Experiment tracking + data versioning:** from day 1, on the empty project.
- **Finetuning:** HF phase — full finetune vs frozen DINOv2 backbone vs LoRA (PEFT). Compare in MLflow.
- **Distributed training:** HF + Accelerate. `accelerate config` runs the *same script* single-device MPS locally and DDP/FSDP multi-GPU on cloud. This pillar requires cloud CUDA — plan a rented GPU session for it.
- **Serving:** wrap inference in FastAPI/BentoML; promote a model from the registry.
- **Monitoring:** Evidently (data/prediction drift) + Prometheus/Grafana (ops). Inject drift deliberately — a clip from a different stadium, lighting, or kit colors — and watch alerts fire. Team classifier is a prime drift target.

---

## 7. Week 1 — concrete TODO

**Goal:** close one full loop — ISSIA → finetuned RT-DETR → tracked annotated video → fully reproducible MLflow run. Ugly results are fine.

### Day 1 — Repo + environment

- [ ] Create skeleton (see §3)
- [ ] Python 3.11 venv/conda env
- [ ] `pip install ultralytics mlflow dvc gdown opencv-python torchmetrics pycocotools`
- [ ] `git init`, commit skeleton
- [ ] Verify MPS: `python -c "import torch; print(torch.backends.mps.is_available())"` → `True`
- [ ] Rule: all params in `configs/*.yaml`

### Day 2 — Instrument before training

- [ ] Start MLflow server; set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`; confirm UI
- [ ] Choose manual `mlflow.start_run()` logging (not autolog)
- [ ] `dvc init`; `dvc remote add -d localremote /tmp/dvc-store`

### Day 3 — Data + COCO converter (the fiddly day)

- [ ] Download ISSIA: `pip install -U gdown && gdown --fuzzy "<drive-url>" -O data/issia.zip`
- [ ] Verify it's a real zip: `file data/issia.zip` → "Zip archive data" (if HTML/tiny, it's the Drive warning page — see §8)
- [ ] `unzip data/issia.zip -d data/issia_raw`; inspect layout: `ls -R data/issia_raw | head`
- [ ] Read FootAndBall's ISSIA loader to learn the GT format
- [ ] Write `src/convert_issia.py` → **COCO JSON** (one camera, sample ~every 5th frame)
- [ ] Write `src/coco_to_yolo.py` (or use Ultralytics `convert_coco`)
- [ ] Wire `dvc.yaml`: raw → coco → yolo as stages; `dvc repro`; commit `dvc.lock`; `dvc push`

### Day 4 — Finetune RT-DETR, logged

- [ ] **Smoke test:** 1 epoch, tiny subset, `device="mps"` — confirm GPU use, watch for "falling back to CPU" warnings. If it crawls → YOLOv8n fallback (pipeline unchanged).
- [ ] Real run: `from ultralytics import RTDETR` → `.train(data="data/issia_yolo/data.yaml", epochs=20, imgsz=640, device="mps")`
- [ ] Wrap in `mlflow.start_run()`: log params (model, epochs, imgsz, data version), `mAP50`, and `best.pt` artifact

### Day 5 — Close the loop visually

- [ ] `model.predict("clip.mp4", save=True)`
- [ ] `model.track("clip.mp4", tracker="bytetrack.yaml", save=True)` → annotated video = deliverable
- [ ] Skim output: expect poor **ball** detection (normal); players should look reasonable

### End-of-week gate

- [ ] **Can you reproduce the run from its MLflow entry + DVC data hash alone?** If yes, the skeleton is sound.

**Explicitly out of scope week 1:** HuggingFace trainer, distributed training, cloud, homography, teams. Those are later phases.

---

## 8. Data strategy & gotchas

- **ISSIA limits:** single ~2-min clip, player/ball/referee boxes, **no player identities, no event labels**. Great for M1–M4 (tracking, homography, physical stats, teams). Cannot support pass accuracy or cross-match player diagnosis — bring SoccerNet/StatsBomb for M5–M6.
- **gdown/Drive failure:** large files often download the "can't virus-scan" HTML page instead of the zip (→ "End-of-central-directory signature not found"). Fix: `pip install -U gdown` + `--fuzzy`. If "quota exceeded," make your own Drive copy and gdown that, or download in browser.
- **Validation philosophy:** many derived stats have no ground-truth labels. ISSIA position GT validates trajectories/speed. Pass stats need StatsBomb. Teams validated visually. "Can I trust this metric without a label?" *is* the monitoring muscle this project builds.

---

## 9. Reference implementations

- Roboflow `sports` (soccer): pitch keypoints → homography → radar → speed/distance — https://github.com/roboflow/sports/tree/main/examples/soccer
- End-to-end football analysis (tracking + homography + speed): https://github.com/AhmedQassemDev2004/computer-vision-football-analysis
- FootAndBall (ISSIA loader / format): https://github.com/jac99/FootAndBall
- Camera calibration in sports: https://blog.roboflow.com/camera-calibration-sports-computer-vision/
- Ultralytics RT-DETR docs: https://docs.ultralytics.com/models/rtdetr
