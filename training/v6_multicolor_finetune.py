"""
v6_multicolor_finetune.py

Current training pipeline: fine-tunes the previously deployed checkpoint
(v4) on a combined black + white connector dataset, so the model handles
both colors without regressing on black (both colors persist in production
— see training/README.md for why this is additive, not a replacement).

Intended to be run as Kaggle notebook cells (split at the "# %% Cell" markers
below if pasting into separate cells). Read training/README.md before
running this, especially the sections on the previous-checkpoint path and
color tagging.
"""

# %% Cell 1: Environment Setup & Dataset Download
# ---------------------------------------------------------------------
# !pip install -q -U ultralytics roboflow

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("sohamblulabel")  # re-add this secret if it's a fresh notebook

import os
import re
import shutil
from collections import Counter

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from roboflow import Roboflow

ultralytics.checks()

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. Go to Notebook Settings -> Accelerator -> GPU T4 x2, then restart."
    )
print(f"GPU available: {torch.cuda.get_device_name(0)}")

rf = Roboflow(api_key=secret_value_0)
project = rf.workspace("soham-bhattacharya-mwcvr").project("bluarmor-od-wd3wq")
version = project.version(3)  # update this version number for future dataset revisions
dataset = version.download("yolov11")
print(f"\nDataset downloaded to: {dataset.location}")


# %% Cell 2: data.yaml Path Correction
# ---------------------------------------------------------------------
DATASET_PATH = dataset.location
data_yaml_path = os.path.join(DATASET_PATH, "data.yaml")

with open(data_yaml_path, "r") as f:
    data_config = yaml.safe_load(f)

fixed_config = dict(data_config)
for split_key, folder_name in [("train", "train"), ("val", "valid"), ("test", "test")]:
    img_dir = os.path.join(DATASET_PATH, folder_name, "images")
    if os.path.isdir(img_dir):
        fixed_config[split_key] = img_dir
fixed_config["path"] = DATASET_PATH

fixed_yaml_path = "/kaggle/working/data.yaml"
with open(fixed_yaml_path, "w") as f:
    yaml.dump(fixed_config, f, default_flow_style=False)

class_names = fixed_config.get("names", [])
print(f"Classes detected: {class_names}")


# %% Cell 2a: Pool All Images and Tag Color + Class
# ---------------------------------------------------------------------
# IMPORTANT: verify the printed group counts below match expectations
# (e.g. ~100/100 for white-defective/white-passing) before continuing.
# If they look wrong, the naming pattern in get_color_for_image() needs
# adjusting to match your actual export — see training/README.md.

def gather_pairs(img_dir):
    label_dir = img_dir.replace("images", "labels")
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        pairs.append((img_path, lbl_path, fname))
    return pairs

all_pairs = []
for split_key in ["train", "val", "test"]:
    if split_key in fixed_config:
        all_pairs.extend(gather_pairs(fixed_config[split_key]))
print(f"Total images pooled: {len(all_pairs)}")

def get_class_for_image(lbl_path):
    """Assumes one connector class per image, per this project's labeling scheme."""
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return None
    with open(lbl_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return None
    class_idx = int(lines[0].split()[0])
    return class_names[class_idx] if 0 <= class_idx < len(class_names) else None

def get_color_for_image(fname):
    """White-connector images in this batch are named pass_NNNN / defect_NNNN.
    Everything else is treated as black. Verify this against your actual
    export before trusting it (see training/README.md)."""
    base = fname.lower()
    if re.match(r'^(pass|defect)_\d+', base):
        return "white"
    return "black"

tagged_pairs = []
for img_path, lbl_path, fname in all_pairs:
    cls = get_class_for_image(lbl_path)
    if cls is None:
        continue
    tagged_pairs.append((img_path, lbl_path, cls, get_color_for_image(fname)))

group_counts = Counter((cls, color) for _, _, cls, color in tagged_pairs)
print("Group counts (class, color) — verify these match expectations:")
for group, count in sorted(group_counts.items()):
    print(f"  {group}: {count}")


# %% Cell 2b: Stratified Re-Split (70/20/10, by class + color)
# ---------------------------------------------------------------------
# Note: this keeps the full black-connector pool rather than re-undersampling
# it. Since this is a fine-tune of an already-strong checkpoint, the
# priority is reinforcing black performance while teaching white — further
# shrinking black-passing data would work against that.

from sklearn.model_selection import train_test_split

group_key = [f"{cls}_{color}" for _, _, cls, color in tagged_pairs]

train_pairs, temp_pairs = train_test_split(
    tagged_pairs, test_size=0.30, stratify=group_key, random_state=42
)
temp_group_key = [f"{cls}_{color}" for _, _, cls, color in temp_pairs]
test_pairs, val_pairs = train_test_split(
    temp_pairs, test_size=(10 / 30), stratify=temp_group_key, random_state=42
)

print(f"Train: {len(train_pairs)} | Test (20%): {len(test_pairs)} | Val (10%): {len(val_pairs)}")
for name, pairs in [("train", train_pairs), ("test", test_pairs), ("val", val_pairs)]:
    counts = Counter((c, col) for _, _, c, col in pairs)
    print(f"  {name}: {dict(counts)}")


# %% Cell 2c: Zoom Preprocessing
# ---------------------------------------------------------------------
# Same ZOOM_FACTOR as the deployed model — geometry hasn't changed.
# Color is carried forward in the output filename (color__basename.jpg)
# since Roboflow's export naming can't be trusted to preserve it otherwise.

ZOOMED_ROOT = "/kaggle/working/zoomed_dataset"
ZOOM_FACTOR = 2.0
MAX_DIM = 1280
MIN_BOX_FRACTION = 0.15

if os.path.exists(ZOOMED_ROOT):
    shutil.rmtree(ZOOMED_ROOT)  # clear stale output — this is what fixed the v3 leakage incident

def adjust_and_write_labels(lbl_path, out_lbl_path, orig_w, orig_h, crop_w, crop_h, x0, y0, min_frac):
    if not os.path.exists(lbl_path):
        open(out_lbl_path, "w").close()
        return
    with open(lbl_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    out_lines = []
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        abs_xc, abs_yc = xc * orig_w, yc * orig_h
        abs_bw, abs_bh = bw * orig_w, bh * orig_h
        x1, y1 = abs_xc - abs_bw / 2, abs_yc - abs_bh / 2
        x2, y2 = abs_xc + abs_bw / 2, abs_yc + abs_bh / 2

        orig_area = (x2 - x1) * (y2 - y1)
        x1c, y1c = max(x1 - x0, 0), max(y1 - y0, 0)
        x2c, y2c = min(x2 - x0, crop_w), min(y2 - y0, crop_h)

        if x2c <= x1c or y2c <= y1c:
            continue
        if (x2c - x1c) * (y2c - y1c) / orig_area < min_frac:
            continue

        new_bw, new_bh = (x2c - x1c) / crop_w, (y2c - y1c) / crop_h
        new_xc, new_yc = (x1c + x2c) / 2 / crop_w, (y1c + y2c) / 2 / crop_h
        out_lines.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

    with open(out_lbl_path, "w") as f:
        f.write("\n".join(out_lines))

for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
    img_out_dir = os.path.join(ZOOMED_ROOT, split_name, "images")
    lbl_out_dir = os.path.join(ZOOMED_ROOT, split_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    for img_path, lbl_path, cls, color in sorted(pairs, key=lambda p: p[0]):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        crop_w, crop_h = w / ZOOM_FACTOR, h / ZOOM_FACTOR
        x0, y0 = (w - crop_w) / 2, (h - crop_h) / 2
        cropped = img[int(y0):int(y0 + crop_h), int(x0):int(x0 + crop_w)]

        scale = min(1.0, MAX_DIM / max(cropped.shape[:2]))
        if scale < 1.0:
            cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{color}__{base_name}"
        cv2.imwrite(os.path.join(img_out_dir, out_name + ".jpg"), cropped)
        adjust_and_write_labels(
            lbl_path, os.path.join(lbl_out_dir, out_name + ".txt"),
            w, h, crop_w, crop_h, x0, y0, MIN_BOX_FRACTION
        )

zoomed_yaml_path = "/kaggle/working/zoomed_data.yaml"
zoomed_config = {
    "path": ZOOMED_ROOT,
    "train": os.path.join(ZOOMED_ROOT, "train", "images"),
    "val": os.path.join(ZOOMED_ROOT, "val", "images"),
    "test": os.path.join(ZOOMED_ROOT, "test", "images"),
    "nc": len(class_names),
    "names": class_names,
}
with open(zoomed_yaml_path, "w") as f:
    yaml.dump(zoomed_config, f, default_flow_style=False)
print(f"Zoomed dataset written to: {zoomed_yaml_path}")


# %% Cell 2f: 3x Offline Augmentation (train split only)
# ---------------------------------------------------------------------
# Same policy as every prior iteration: no flips (connector is polarized/
# asymmetric), corrected GaussNoise API (std_range, not the deprecated
# var_limit), and explicit int() casting on class labels before writing
# (Albumentations returns them as floats internally).

import albumentations as A

augment_pipeline = A.Compose([
    A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    # no flips — connector is polarized/asymmetric
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.4))

train_img_dir = os.path.join(ZOOMED_ROOT, "train", "images")
train_lbl_dir = os.path.join(ZOOMED_ROOT, "train", "labels")

for fname in list(os.listdir(train_img_dir)):
    if "_aug" in fname:
        os.remove(os.path.join(train_img_dir, fname))
for fname in list(os.listdir(train_lbl_dir)):
    if "_aug" in fname:
        os.remove(os.path.join(train_lbl_dir, fname))

base_images = sorted(f for f in os.listdir(train_img_dir) if "_aug" not in f)

for fname in base_images:
    base_name = os.path.splitext(fname)[0]
    img = cv2.imread(os.path.join(train_img_dir, fname))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lbl_path = os.path.join(train_lbl_dir, base_name + ".txt")
    bboxes, class_labels = [], []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    class_labels.append(int(float(parts[0])))
                    bboxes.append([float(x) for x in parts[1:5]])

    for aug_idx in range(2):
        augmented = augment_pipeline(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
        aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        out_name = f"{base_name}_aug{aug_idx}"
        cv2.imwrite(os.path.join(train_img_dir, out_name + ".jpg"), aug_img)
        with open(os.path.join(train_lbl_dir, out_name + ".txt"), "w") as f:
            for cls, box in zip(augmented["class_labels"], augmented["bboxes"]):
                f.write(f"{int(cls)} {' '.join(f'{v:.6f}' for v in box)}\n")

print(f"Train images after 3x augmentation: {len(os.listdir(train_img_dir))}")


# %% Leak-Check Gate — must print zero overlap before training
# ---------------------------------------------------------------------

def get_source_id(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r"_aug\d+$", "", name)

split_ids = {
    split: set(get_source_id(f) for f in os.listdir(os.path.join(ZOOMED_ROOT, split, "images")))
    for split in ["train", "val", "test"]
}
overlaps = {f"{a}/{b}": split_ids[a] & split_ids[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}

if any(overlaps.values()):
    for pair, ov in overlaps.items():
        if ov:
            print(f"LEAKAGE in {pair}: {len(ov)} overlapping images")
    raise RuntimeError("DATA LEAKAGE DETECTED — refusing to proceed to training.")
print("No leakage detected — safe to proceed to training.")


# %% Cell 4: Fine-Tune From the Previously Deployed Checkpoint
# ---------------------------------------------------------------------
# Upload the previous best.pt as a Kaggle Model/Dataset first, then verify
# the exact mounted path with os.listdir("/kaggle/input") — do not assume
# it matches the placeholder below.

PREVIOUS_BEST_PATH = "/kaggle/input/models/<username>/<model-name>/pytorch/default/1/best.pt"

# Sanity check before training — confirm this is actually the fine-tuned
# checkpoint (defective/passing), not a fresh COCO-pretrained model:
_check = YOLO(PREVIOUS_BEST_PATH)
print(_check.names)  # should print {0: 'defective', 1: 'passing'}

model = YOLO(PREVIOUS_BEST_PATH)
results = model.train(
    data=zoomed_yaml_path,
    imgsz=640,
    epochs=40,
    batch=16,
    patience=15,
    lr0=0.002,        # lower than a from-scratch run — this is a fine-tune
    device=0,
    project="/kaggle/working/runs",
    name="aoi_pass_defect_v6_multicolor",
    exist_ok=True,
    save=True,
    save_period=10,
    plots=True,
    verbose=True,
    seed=42,
    fliplr=0.0,
)
print("Best weights: /kaggle/working/runs/aoi_pass_defect_v6_multicolor/weights/best.pt")


# %% Cell 5: Evaluate Overall, Then Split by Color
# ---------------------------------------------------------------------
# This is the critical regression check: report per-color, per-class AP50,
# not just an overall blended number, to catch any regression on either
# connector color introduced by this fine-tune.

best_model = YOLO("/kaggle/working/runs/aoi_pass_defect_v6_multicolor/weights/best.pt")
print(f"Loaded checkpoint: {best_model.ckpt_path}")  # confirm this is v6, not a leftover v4 reference

test_metrics = best_model.val(data=zoomed_yaml_path, imgsz=640, device=0, split="test")
print("=== Overall Test Metrics ===")
print(f"mAP50: {test_metrics.box.map50:.4f} | mAP50-95: {test_metrics.box.map:.4f}")
for i, cname in enumerate(class_names):
    print(f"  {cname}: AP50 = {test_metrics.box.ap50[i]:.4f}")

def build_color_subset_yaml(color):
    src_img_dir = os.path.join(ZOOMED_ROOT, "test", "images")
    src_lbl_dir = os.path.join(ZOOMED_ROOT, "test", "labels")
    subset_root = f"/kaggle/working/test_{color}_only"
    if os.path.exists(subset_root):
        shutil.rmtree(subset_root)
    img_out, lbl_out = os.path.join(subset_root, "images"), os.path.join(subset_root, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for fname in os.listdir(src_img_dir):
        if fname.lower().startswith(f"{color}__"):
            shutil.copy(os.path.join(src_img_dir, fname), img_out)
            lbl_src = os.path.join(src_lbl_dir, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, lbl_out)
    subset_yaml = f"{subset_root}.yaml"
    with open(subset_yaml, "w") as f:
        yaml.dump({"path": subset_root, "train": img_out, "val": img_out, "test": img_out,
                   "nc": len(class_names), "names": class_names}, f)
    return subset_yaml, img_out

for color in ["black", "white"]:
    subset_yaml, img_out = build_color_subset_yaml(color)
    n = len(os.listdir(img_out))
    print(f"\n=== {color.upper()}-only test subset ({n} images) ===")
    if n == 0:
        print("  No images found — recheck the color heuristic in Cell 2a.")
        continue
    m = best_model.val(data=subset_yaml, imgsz=640, device=0, split="test")
    print(f"mAP50: {m.box.map50:.4f}")
    for i, cname in enumerate(class_names):
        if i < len(m.box.ap50):
            print(f"  {cname}: AP50 = {m.box.ap50[i]:.4f}")


# %% Cell 6: Packaging
# ---------------------------------------------------------------------
output_dir = "/kaggle/working/runs/aoi_pass_defect_v6_multicolor"
shutil.make_archive("/kaggle/working/aoi_pass_defect_v6_multicolor_results", "zip", output_dir)
print("Download from Kaggle 'Output' tab: aoi_pass_defect_v6_multicolor_results.zip")
