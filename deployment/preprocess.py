"""
preprocess.py

The zoom_crop() transform used both at training time (Kaggle) and at
inference time (this Pi deployment). These MUST match exactly — feeding
the deployed model full, uncropped camera frames presents the connector
at a different effective scale than the model was trained on, independent
of detection accuracy.

If ZOOM_FACTOR or MAX_DIM ever change during a future training iteration
(e.g. after a fixture/working-distance change), update both here and in
the corresponding Kaggle preprocessing cell together — do not let them
drift out of sync.
"""

import cv2

ZOOM_FACTOR = 2.0
MAX_DIM = 1280


def zoom_crop(frame):
    """
    Center-crop to 2x zoom (central 50% width/height), then downsize only
    if the result exceeds MAX_DIM on the longest side — never upscale.
    Upscaling a smaller pixel region does not add real detail and actively
    discards sharpness (this was an early bug in the training pipeline).
    """
    h, w = frame.shape[:2]
    crop_w, crop_h = w / ZOOM_FACTOR, h / ZOOM_FACTOR
    x0, y0 = (w - crop_w) / 2, (h - crop_h) / 2
    x1, y1 = x0 + crop_w, y0 + crop_h
    cropped = frame[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]

    ch, cw = cropped.shape[:2]
    scale = min(1.0, MAX_DIM / max(ch, cw))
    if scale < 1.0:
        out_w, out_h = int(round(cw * scale)), int(round(ch * scale))
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return cropped
