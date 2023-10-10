import os
import re
import sys
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import multiprocessing as mp
import traceback
import shutil
from termcolor import colored

# W, H
BBOX_SIZE = np.array([256, 256], dtype=np.int32)
MARGIN_SIZE = np.array([.5, .5])
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
    signal_icons: List[np.ndarray],
    light_state,
    vehicle,
    weather,
):
    def _print(msg):
        w = shutil.get_terminal_size()[0]
        msg = f"{frame_number:3d}: {msg}"
        if len(msg) < w:
            msg += " " * (w - len(msg))
        print(msg, end="\r", flush=True)

    ################################
    # Find pixels occupied by car
    ################################

    mask = (sem[:, :, 0] >= min(CAR_SEM_VALS)) & (sem[:, :, 0] <= max(CAR_SEM_VALS))
    light_mask = sem[:, :, 0] == 32
    mask |= light_mask

    if not mask.any():
        _print("No car detected, skipping")
        return

    if not light_mask.any():
        _print("No lights detected, skipping")
        return

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
    light_mask = light_mask[t:b, l:r]

    ################################
    # Calculate light activations
    ################################

    light_ids = np.round(np.where(light_mask, sem_bboxed[:, :, 2] / 255 * 16, -1)).astype(
        np.int8
    )
    light_activations = sem_bboxed[:, :, 1] / 255

    def _get_score(light_id):
        return np.max(np.where(light_ids == light_id, light_activations, -1))

    # brake (Y=8+4), left (R=8), right (M=8+2)
    light_scores = [_get_score(12), _get_score(8), _get_score(10)]

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

    dst = Image.new(
        "RGB", (cam_img.width + sem_img.width + cam_img.height // 3, cam_img.height)
    )
    dst.paste(cam_img, (0, 0))
    dst.paste(sem_img, (cam_img.width, 0))

    # Label lights not found in the mask as U (unknown), lights that are off in this frame as O (off),
    # otherwise use label from the light_state
    current_light_state = "".join(
        [
            "U" if s < 0 else ("O" if s < 0.5 else c)
            for c, s in zip(light_state, light_scores)
        ]
    )

    for i, (state, icon) in enumerate(zip(current_light_state, signal_icons)):
        if state == "U":
            continue
        if state == "O":
            icon //= 3

        dst.paste(Image.fromarray(icon), (cam_img.width * 2, (i * cam_img.height) // 3))

    ################################
    # Write output
    ################################

    filename = [
        f"{frame_number:04d}",
        vehicle,
        weather,
        "back",
        light_state,
        current_light_state,
        f"l{margin_l / scale_w:03.0f}r{margin_r / scale_w:03.0f}t{margin_t / scale_h:03.0f}b{margin_b / scale_h:03.0f}",
    ]
    filename = "_".join([re.sub(r"\W|_", ".", str(s)) for s in filename]) + '.png'
    dst.save(os.path.join(path, filename))

    def _colored(light_state_str):
        b = light_state_str[0]
        l = light_state_str[1]
        r = light_state_str[2]
        b = colored(b, "light_red" if b == "B" else ("red" if b == "O" else "dark_grey"))
        l = colored(
            l, "light_yellow" if l == "L" else ("yellow" if l == "O" else "dark_grey")
        )
        r = colored(
            r, "light_yellow" if r == "R" else ("yellow" if r == "O" else "dark_grey")
        )

        return b + l + r

    _print(f"Saved: lbl={_colored(light_state)} cur={_colored(current_light_state)}")


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
        self.pool = mp.Pool(12, maxtasksperchild=10)

        mask_sz = OUTPUT_SIZE[1] // 3
        self.signal_masks = [
            Image.open(f"/home/carla/Scenic/assets/img/{name}.png")
            for name in ("brake", "left", "right")
        ]
        self.signal_masks = [
            mask.resize((mask_sz, mask_sz), Image.Resampling.BILINEAR)
            for mask in self.signal_masks
        ]
        self.signal_masks = [np.array(mask) for mask in self.signal_masks]

        # Ordered BLR
        mask_colors = [np.array([1.0, 0, 0])] + [np.array([1, 0.867, 0])] * 2
        self.signal_masks = [
            self._color_mask(mask, color)
            for mask, color in zip(self.signal_masks, mask_colors)
        ]

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

    def _color_mask(self, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
        colored_mask = mask[:, :, 3:4] * np.broadcast_to(color, (1, 1, color.shape[0]))
        return colored_mask.astype(np.uint8)

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
                self.signal_masks,
                self._light_state,
                self._vehicle,
                self._weather,
            ),
            error_callback=_supress_error_print,
        )
