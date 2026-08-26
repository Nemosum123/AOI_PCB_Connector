"""
touchscreen_app.py

Fullscreen kiosk UI for the 7" (1024x600) touchscreen.

Imports inspection_core so capture / inference / logging behavior is
identical to the validated pipeline. Do not reimplement any of that
logic here.

Run from the graphical session (not a bare SSH shell with no display
context):
    DISPLAY=:0 python3 touchscreen_app.py
"""

import queue
import threading
import tkinter as tk

import cv2
from PIL import Image, ImageTk

import inspection_core as core

# Single-station rig — change here if the active camera ever needs to change.
CAMERA_INDEX = 0

SCREEN_W, SCREEN_H = 1024, 600
IMAGE_PANEL_W, IMAGE_PANEL_H = 620, 540

VERDICT_COLORS = {
    "PASS": "#1e8e3e",
    "FAIL": "#d93025",
    "FLAG_FOR_REVIEW": "#f9ab00",
}


class AOIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BluArmor AOI")
        self.root.geometry(f"{SCREEN_W}x{SCREEN_H}")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#111111")

        self.result_queue = queue.Queue()
        self.busy = False

        self._build_layout()

        self.picam2 = core.init_camera(CAMERA_INDEX)
        core.load_model()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_queue)

    def _build_layout(self):
        # Left: image panel
        self.image_label = tk.Label(self.root, bg="#000000")
        self.image_label.place(x=10, y=10, width=IMAGE_PANEL_W, height=IMAGE_PANEL_H)

        panel_x = IMAGE_PANEL_W + 30

        self.capture_btn = tk.Button(
            self.root,
            text="CAPTURE",
            font=("Helvetica", 28, "bold"),
            bg="#1a73e8",
            fg="white",
            activebackground="#1558b0",
            command=self.on_capture,
        )
        self.capture_btn.place(x=panel_x, y=20, width=340, height=150)

        self.result_label = tk.Label(
            self.root,
            text="Ready",
            font=("Helvetica", 32, "bold"),
            bg="#333333",
            fg="white",
            wraplength=340,
            justify="center",
        )
        self.result_label.place(x=panel_x, y=200, width=340, height=150)

        self.confidence_label = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 16),
            bg="#111111",
            fg="#cccccc",
        )
        self.confidence_label.place(x=panel_x, y=360, width=340, height=40)

        self.quit_btn = tk.Button(
            self.root, text="Quit", font=("Helvetica", 12), command=self.on_close
        )
        self.quit_btn.place(x=panel_x, y=540, width=100, height=40)

    def on_capture(self):
        if self.busy:
            return  # ignore taps while an inspection is already running
        self.busy = True
        self.capture_btn.config(state="disabled", bg="#888888")
        self.result_label.config(text="Processing...", bg="#333333")
        self.confidence_label.config(text="")

        threading.Thread(target=self._run_inspection_thread, daemon=True).start()

    def _run_inspection_thread(self):
        try:
            verdict, confidence, frame = core.run_inspection(self.picam2, CAMERA_INDEX)
            self.result_queue.put(("ok", verdict, confidence, frame))
        except Exception as e:
            self.result_queue.put(("error", str(e), None, None))

    def _poll_queue(self):
        try:
            status, a, b, c = self.result_queue.get_nowait()
            if status == "ok":
                self._show_result(a, b, c)
            else:
                self.result_label.config(text="ERROR", bg="#d93025")
                self.confidence_label.config(text=a)
            self.busy = False
            self.capture_btn.config(state="normal", bg="#1a73e8")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_result(self, verdict, confidence, frame_bgr):
        color = VERDICT_COLORS.get(verdict, "#333333")
        self.result_label.config(text=verdict.replace("_", " "), bg=color)
        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}" if confidence is not None else "No detection"
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((IMAGE_PANEL_W, IMAGE_PANEL_H))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # keep a reference or Tk will garbage-collect it

    def on_close(self):
        core.release_camera(self.picam2)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AOIApp(root)
    root.mainloop()
