import os
import re
import sys
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import multiprocessing as mp
import traceback
import shutil
import pandas as pd
from termcolor import colored

# W, H
BBOX_SIZE = np.array([256, 256], dtype=np.int32)
MARGIN_SIZE = np.array([0.5, 0.5])
OUTPUT_SIZE = (BBOX_SIZE * (1 + 2 * MARGIN_SIZE)).astype(np.int32)
BBOX_MIN_SIZE = (100, 100)

CAR_SEM_VALS = [
    13,  # rider
    14,  # Car
    15,  # truck
    16,  # bus
    17,  # train
    18,  # motorcycle
    19,  # bicycle
]

LIGHT_SEM_VAL = 32

# The numeric IDs stored in semantic segmentation blue channel (0-255) are mapped to (0-15)
LIGHT_IDS = {
    "low_beam": 0,
    "high_beam": 2,
    "reverse": 4,
    "front_fog": 6,
    "rear_fog": 7,
    "left_ind": 8,
    "right_ind": 10,
    "brake": 12,
    "front_pos": 14,
    "rear_pos": 15,
}


def _safe_ingest_async(*args, **kwargs):
    try:
        _ingest_async(*args, **kwargs)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)


def _ingest_async(
    cam: np.ndarray,
    sem: np.ndarray,
    path: str,
    frame_number: int,
    light_state,
    vehicle,
    weather,
):
    def _print(msg):
        w = shutil.get_terminal_size()[0]
        msg = f"\r{frame_number:3d}: {msg}"
        if len(msg) < w:
            msg += " " * (w - len(msg))
        print(msg, end="", flush=True)

    ################################
    # Find pixels occupied by car
    ################################

    light_mask = sem[:, :, 0] == 32

    if not light_mask.any():
        _print("No lights detected, skipping")
        return

    mask = (sem[:, :, 0] >= min(CAR_SEM_VALS)) & (sem[:, :, 0] <= max(CAR_SEM_VALS))
    mask |= light_mask

    ################################
    # Calculate bounding box
    ################################

    h = np.nonzero(mask.sum(axis=1))
    v = np.nonzero(mask.sum(axis=0))

    t, b = h[0][0], h[0][-1] + 1
    l, r = v[0][0], v[0][-1] + 1

    if (b - t) < BBOX_MIN_SIZE[0] or (r - l) < BBOX_MIN_SIZE[1]:
        _print(f"Too small ({b - t :3d}, {r - l :3d}) < {BBOX_MIN_SIZE}, skipping")

    ################################
    # Crop masks / images to bbox
    ################################

    mask = mask[t:b, l:r]
    mask = np.stack([mask] * 3, axis=2)

    sem_bboxed = sem[t:b, l:r, :]
    cam_bboxed = cam[t:b, l:r, :]
    light_mask = light_mask[t:b, l:r]

    ################################
    # Calculate light activations
    ################################

    light_ids = np.round(np.where(light_mask, sem_bboxed[:, :, 2] / 255 * 16, -1)).astype(
        np.int8
    )
    light_activations = sem_bboxed[:, :, 1] / 255

    def _get_score(light_name: str):
        light_id = LIGHT_IDS[light_name]
        id_mask = light_ids == light_id
        return np.max(np.where(id_mask, light_activations, -1))

    light_scores = {k: _get_score(k) for k in LIGHT_IDS.keys()}

    ################################
    # Resize to 224x224, assemble
    ################################

    bbox_w = r - l
    bbox_h = b - t

    margin_t = int(min(MARGIN_SIZE[0] * bbox_h, t))
    margin_b = int(min(MARGIN_SIZE[0] * bbox_h, cam.shape[0] - b))
    margin_l = int(min(MARGIN_SIZE[1] * bbox_w, l))
    margin_r = int(min(MARGIN_SIZE[1] * bbox_w, cam.shape[1] - r))

    cam_cropped = cam[t - margin_t : b + margin_b, l - margin_l : r + margin_r, :]
    sem_cropped = sem[t - margin_t : b + margin_b, l - margin_l : r + margin_r, :]

    cam_img = Image.fromarray(cam_cropped)
    sem_img = Image.fromarray(sem_cropped)

    cam_img = cam_img.resize(OUTPUT_SIZE)
    sem_img = sem_img.resize(OUTPUT_SIZE)

    scale_h = cam_cropped.shape[0] / OUTPUT_SIZE[0]
    scale_w = cam_cropped.shape[1] / OUTPUT_SIZE[1]

    ################################
    # Write output
    ################################

    metadata = pd.DataFrame(
        {
            "vehicle": vehicle,
            "weather": weather,
            "light_state": light_state,
            "margin_l": int(margin_l / scale_w),
            "margin_r": int(margin_r / scale_w),
            "margin_t": int(margin_t / scale_h),
            "margin_b": int(margin_b / scale_h),
            **{f"score_{name}": score for name, score in light_scores.items()},
        },
        index=[frame_number],
    )

    filename = [
        f"{frame_number:04d}",
        vehicle,
        weather,
        light_state,
    ]
    filename = "_".join([re.sub(r"\W|_", ".", str(s)) for s in filename])
    cam_img.save(os.path.join(path, filename + ".jpg"), quality=95)
    sem_img.save(os.path.join(path, "mask_" + filename + ".png"), compress_level=9)

    csv_filename = os.path.join(path, "metadata.csv")
    if os.path.exists(csv_filename):
        metadata.to_csv(csv_filename, mode="a", header=False)
    else:
        metadata.to_csv(csv_filename)

    _print(f"Saved: {vehicle} lbl={light_state}")


class DataRecorder:
    def __init__(self, weather=None, light_state=None, vehicle=None) -> None:
        self._recordings_path = "../recordings"
        os.makedirs(self._recordings_path, exist_ok=True)
        scenario_number = len(os.listdir(self._recordings_path))
        self.path = os.path.abspath(
            os.path.join(self._recordings_path, f"{scenario_number:07d}")
        )
        path = self.path
        i = 1
        while os.path.exists(path):
            path = f"{self.path}-{i:03d}"
            i += 1

        self.path = path
        os.makedirs(self.path)

        self.frame_number = -1
        self.pool = mp.Pool(12)

        self._weather = None
        self._light_state = None
        self._vehicle = None

        if weather is not None:
            self.set_weather(weather)
        if light_state is not None:
            self.set_light_state(light_state)
        if vehicle is not None:
            self.set_vehicle(vehicle)

    def set_weather(self, weather: str):
        self._weather = weather

    def set_light_state(self, light_state: Tuple[bool, bool, bool]):
        b, l, r = light_state
        self._light_state = (
            ("B" if b else "O") + ("L" if l else "O") + ("R" if r else "O")
        )

    def set_vehicle(self, vehicle: str):
        if vehicle.startswith("vehicle."):
            vehicle = vehicle[len("vehicle.") :]
        self._vehicle = vehicle

    def ingest(self, cam: Optional[np.ndarray], sem: Optional[np.ndarray]):
        self.frame_number += 1

        if cam is None or sem is None:
            return

        if self._vehicle is None or self._light_state is None or self._weather is None:
            return

        def _supress_error_print(e: BaseException):
            if isinstance(e, KeyboardInterrupt):
                self.pool.terminate()
            else:
                traceback.print_exception(type(e), e, e.__traceback__)

        self.pool.apply_async(
            _safe_ingest_async,
            (
                cam,
                sem,
                self.path,
                self.frame_number,
                self._light_state,
                self._vehicle,
                self._weather,
            ),
            error_callback=_supress_error_print,
        )
