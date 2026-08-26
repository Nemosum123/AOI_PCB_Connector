# BluArmor AOI — PCB Connector Alignment Detection

Edge-based Automated Optical Inspection (AOI) system that replaces manual visual
inspection/logging of PCB battery connector alignment on a BluArmor (Bangalore)
production line. A YOLO11n object detection model runs entirely on-device on a
Raspberry Pi 5, driven by a touchscreen UI for line technicians.

**Status:** Deployed proof-of-concept. Running with human spot-checking during
an initial production trial — see [Known Issues & Roadmap](#known-issues--roadmap)
before treating this as fully autonomous.

## Table of Contents
- [Overview](#overview)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Model History](#model-history)
- [Setup](#setup)
- [Training Pipeline](#training-pipeline)
- [Deployment](#deployment)
- [Touchscreen UI](#touchscreen-ui)
- [Known Issues & Roadmap](#known-issues--roadmap)
- [License](#license)

## Overview

The system captures an image of a PCB battery connector, crops and preprocesses
it to match the model's training-time framing, runs it through a fine-tuned
YOLO11n detector, and returns one of three verdicts:

- **PASS** — a `passing` (correctly aligned) connector was detected
- **FAIL** — a `defective` (misaligned) connector was detected
- **FLAG_FOR_REVIEW** — no confident detection either way (not automatically
  treated as safe, since this is a two-class scheme)

Every inspection is logged to CSV, and non-PASS results save an annotated
image for audit purposes.

## Hardware

- Raspberry Pi 5, 4GB RAM, Raspberry Pi OS (Bookworm)
- 2x Raspberry Pi Camera Module 3 (Sony IMX708), connected via the Pi 5's
  native dual CSI ports (CSI0/CSI1) — no external multiplexer required
- LED ring light, fixed elevated desk-mount rig
- 7" HDMI touchscreen (1024x600 native resolution), touch via USB-A-to-B
  (WaveShare WS170120 panel), driven through XWayland on the default
  Raspberry Pi OS Bookworm desktop session

## Repository Structure

```
bluarmor-aoi/
├── README.md                          # this file
├── LICENSE
├── requirements-pi.txt                # deployment (Raspberry Pi) dependencies
├── requirements-training.txt          # training (Kaggle) dependencies
├── deployment/
│   ├── preprocess.py                  # zoom_crop() — must match training-time transform exactly
│   ├── inspection_core.py             # capture -> preprocess -> inference -> verdict -> logging
│   ├── touchscreen_app.py             # Tkinter kiosk UI for the 7" touchscreen
│   └── models/                        # NCNN-exported model goes here (not committed — see below)
├── training/
│   ├── README.md                      # how to run the Kaggle pipeline
│   └── v6_multicolor_finetune.py      # current training pipeline (fine-tune, black+white connectors)
├── docs/
│   ├── ITERATION_HISTORY.md           # v1 → v6 model iteration summary
│   ├── OUTSTANDING_ITEMS.md           # prioritized next steps
│   └── AOI_Final_Report.pdf           # full original project report (iterations 1-3 + deployment)
└── logs/                              # inspection_log.csv and flagged_images/ live here at runtime
|__ jupyter_notebook                                       
```

**Model weights (`.pt`, NCNN exports) are intentionally not committed to this
repo.** They're large binaries that don't belong in plain git history — distribute
them via GitHub Releases, Kaggle Output artifacts, or shared storage, and drop
the exported model into `deployment/models/` locally on each device.

## Model History

| Version | Classes | Key change | Test mAP50 | Notes |
|---|---|---|---|---|
| v1 | 1 (`connector`) | Baseline, implicit background | 0.9950 | `fliplr=0.5` left enabled — flagged as a bug given the connector is polarized |
| v2 | 2 (`defective`/`passing`) | Explicit two-class scheme | 0.9949 | Val split severely skewed (13% defective); `fliplr` still unresolved |
| v4 | 2 | Class-balanced, 2x zoom-crop, 3x offline aug, `fliplr=0.0` finally applied | 0.9738 | Final iteration before deployment. Passing-class AP50 softened (0.9948→0.9536) from undersampling trade-off — later caused a live false negative |
| — | 2 | Elevated fixture evaluated | — | Live-validated against existing v4 checkpoint; no retrain performed after stress-testing held up |
| v6 (current) | 2 | Fine-tuned from v4 on combined black+white connector data | 0.9900 overall (0.985 black-defective / 0.995 black-passing / 0.995 white-defective / 0.995 white-passing) | Additive fine-tune — both colors persist in production, so black data was combined with white, not replaced |

Full detail on each iteration, including bugs found/fixed and the data-leakage
incident in v3 (discarded), is in [`docs/ITERATION_HISTORY.md`](docs/ITERATION_HISTORY.md)
and the original [project report](docs/AOI_Final_Report.pdf).

## Setup

### Raspberry Pi (deployment environment)

```bash
python3 -m venv --system-site-packages ~/aoi-env   # --system-site-packages is required
                                                     # for picamera2's system libcamera bindings
source ~/aoi-env/bin/activate

# PyTorch MUST come from the official CPU wheel index — default aarch64 pip
# resolution can grab a CUDA/Jetson-targeted build that wastes 500MB+ of disk.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-pi.txt

# Pillow's Tk bindings ship as a separate apt package on Raspberry Pi OS —
# plain `pip install pillow` does not include ImageTk.
sudo apt install -y python3-pil.imagetk
```

`source ~/aoi-env/bin/activate` must be re-run every new terminal/SSH session
— it does not persist.

### Kaggle (training environment)

See [`training/README.md`](training/README.md) for the full pipeline. In short:
install from `requirements-training.txt`, store your Roboflow API key in
Kaggle Secrets (never hardcode it — see the note in that file), and upload
the previous checkpoint as a Kaggle Model/Dataset when fine-tuning rather
than training from scratch.

## Training Pipeline

The current training pipeline (`training/v6_multicolor_finetune.py`) fine-tunes
from the previously deployed checkpoint rather than training from scratch,
using a combined black + white connector dataset. Key steps: pool and
color-tag all images → stratified 70/20/10 split by class *and* color →
zoom-crop preprocessing (matches deployment exactly) → 3x offline augmentation
(train split only, no flips) → an automated leak-check gate that must show
zero cross-split overlap → fine-tune → evaluate overall *and* split by color
to catch any regression on either connector color.

See [`training/README.md`](training/README.md) for prerequisites and how to
adapt this for a future iteration.

## Deployment

1. Back up the currently deployed model before replacing it.
2. `scp` the new `best.pt` from Kaggle onto the Pi.
3. Export to NCNN **on the Pi** (not on Kaggle — this is device-specific):
   ```bash
   python3 -c "from ultralytics import YOLO; YOLO('best.pt').export(format='ncnn')"
   ```
4. Confirm class names: `YOLO('best_ncnn_model').names` should print
   `{0: 'defective', 1: 'passing'}`.
5. Place the exported `best_ncnn_model/` folder at the path
   `deployment/inspection_core.py`'s `MODEL_PATH` expects.
6. Confirm `CONF_THRESHOLD` in `inspection_core.py` is at its intended
   production value (`0.4` as of this writing) — not a leftover diagnostic
   value from threshold tuning.
7. Smoke-test with one known-good board of each connector color before
   running a full live validation pass.

## Touchscreen UI

`deployment/touchscreen_app.py` is a fullscreen Tkinter kiosk app sized to a
1024x600 touchscreen: a Capture button, an image panel showing the annotated
result, and a color-coded verdict panel (green PASS / red FAIL / amber
FLAG_FOR_REVIEW). It imports `inspection_core.py` directly rather than
duplicating any inference logic, so its behavior always matches whatever
model and threshold are currently configured.

Run it from the graphical session (not a bare SSH shell with no display context):

```bash
DISPLAY=:0 python3 touchscreen_app.py
```

## Known Issues & Roadmap

See [`docs/OUTSTANDING_ITEMS.md`](docs/OUTSTANDING_ITEMS.md) for the full,
prioritized list. Headline items:

1. **Asymmetric confidence threshold** (top priority, not yet implemented) —
   a missed defect is more costly than a false alarm, so `FAIL` should trigger
   at a lower confidence bar than `PASS` confirms. Currently symmetric at 0.4.
2. Live validation surfaced **1-2 false negatives** (defective board scored
   as PASS) prior to the multi-color fine-tune, consistent with the
   passing-class AP50 softening from v4. Continue running with human
   spot-checking until the asymmetric threshold is in place and re-validated.
3. `SAVE_ALL_IMAGES` toggle exists in `inspection_core.py` for use during
   threshold tuning, defaulting to `False` in normal operation.
4. No storage retention policy yet for `logs/flagged_images/`.
5. Recurring re-validation cadence against a fixed known-board set (now
   covering both connector colors) should be established before trusting
   any future model or threshold change.

## License

All data used for this project belongs to BluArmor
