"""
inspection_core.py

Shared inference module for the BluArmor AOI system.

This is the single source of truth for capture -> zoom_crop -> inference ->
verdict -> logging. Both the touchscreen UI (touchscreen_app.py) and any
future CLI tooling should import this rather than duplicating the logic.
If you change the verdict logic, threshold, or logging format, change it
here once.
"""

import csv
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

from preprocess import zoom_crop

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_PATH = "models/best_ncnn_model"

# TODO (outstanding item #1, top priority): split into separate thresholds
# for FAIL vs PASS once asymmetric confidence thresholding is implemented.
# A missed defect is more costly than a false alarm, so the FAIL threshold
# should end up lower than the PASS threshold. Currently symmetric.
CONF_THRESHOLD = 0.4

LOG_PATH = "logs/inspection_log.csv"
FLAGGED_DIR = "logs/flagged_images"

# TODO (outstanding item #2): flip to True during threshold tuning so every
# inspection leaves a saved image, not just non-PASS results. Leave False
# in normal operation to avoid unbounded storage growth.
SAVE_ALL_IMAGES = False

_model = None


def load_model():
    """Load the NCNN model once. Call at app startup, not per-inspection."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def init_camera(camera_index):
    """
    Initialize a single camera using the known-safe default config.

    IMPORTANT: do not pass a custom resolution/format to
    create_still_configuration(). A custom (1920x1080, RGB888) config
    combined with an immediate autofocus_cycle() previously triggered a
    kernel-level RP1 camera driver fault on this hardware (confirmed via
    dmesg). Default config + settle delay + non-fatal autofocus is the
    validated-working path. All resolution reduction happens in software
    via zoom_crop().
    """
    picam2 = Picamera2(camera_num=camera_index)
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # settle delay — required, do not remove
    try:
        picam2.autofocus_cycle()
    except Exception as e:
        print(f"Autofocus failed (non-fatal): {e}")
    return picam2


def release_camera(picam2):
    picam2.stop()
    picam2.close()


def _ensure_log_dirs():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(FLAGGED_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "camera_index", "verdict", "confidence", "final_verdict"]
            )


def run_inspection(picam2, camera_index):
    """
    Capture one frame and run it through the full pipeline.

    Returns:
        final_verdict (str): "PASS", "FAIL", or "FLAG_FOR_REVIEW"
        confidence (float or None): confidence of the top detection
        display_frame (numpy array, BGR): annotated frame for UI display
    """
    _ensure_log_dirs()
    model = load_model()

    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cropped = zoom_crop(frame_bgr)

    results = model.predict(cropped, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]

    verdict = "NONE"
    confidence = None
    if len(r.boxes) > 0:
        best_idx = int(r.boxes.conf.argmax())
        cls_id = int(r.boxes.cls[best_idx])
        confidence = float(r.boxes.conf[best_idx])
        verdict = model.names[cls_id].upper()  # "DEFECTIVE" or "PASSING"

    if verdict == "DEFECTIVE":
        final_verdict = "FAIL"
    elif verdict == "PASSING":
        final_verdict = "PASS"
    else:
        final_verdict = "FLAG_FOR_REVIEW"

    display_frame = r.plot()  # always annotated, for on-screen display

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            [timestamp, camera_index, verdict, confidence, final_verdict]
        )

    if final_verdict != "PASS" or SAVE_ALL_IMAGES:
        safe_ts = timestamp.replace(":", "-").replace(" ", "_")
        fname = f"{FLAGGED_DIR}/{safe_ts}_{final_verdict}.jpg"
        cv2.imwrite(fname, display_frame)

    return final_verdict, confidence, display_frame
