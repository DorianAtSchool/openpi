"""Policy transforms for Franka pick-and-place dataset (DorianAtSchool/pick_place).

Dataset features:
  - observation.images.top:   video (224, 224, 3) — overhead camera
  - observation.images.side:  video (224, 224, 3) — side camera
  - observation.images.wrist: video (224, 224, 3) — wrist camera
  - observation.state:        float32 [8]  — eef pose (6) + gripper widths (2)
  - action:                   float32 [7]  — delta EEF (6) + gripper (1)

Actions are already delta (EEF velocity), so no extra DeltaActions transform is needed.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_pick_place_example() -> dict:
    """Creates a random input example for the Franka pick-and-place policy."""
    return {
        "observation/top_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/side_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": "pick up the red block and place it on the green target",
    }


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaPickPlaceInputs(transforms.DataTransformFn):
    """Converts inputs from the Franka pick-and-place dataset to the model's expected format.

    This is used for both training (after RepackTransform remaps dataset keys) and inference
    (where the environment directly passes keys in observation/xxx format).

    Pi0/Pi0.5 supports three image slots: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
    We map:
      - observation/top_image  → base_0_rgb      (main third-person view)
      - observation/wrist_image → left_wrist_0_rgb (wrist camera)
      - observation/side_image  → right_wrist_0_rgb (second third-person view — re-using the slot)

    Since pi0.5 masks out right_wrist for PI0 type but not PI0_FAST, we set all masks to True
    because we actually have a real image in the right_wrist slot.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top_image = _parse_image(data["observation/top_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        side_image = _parse_image(data["observation/side_image"])

        state = np.asarray(data["observation/state"], dtype=np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": side_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We have a real side camera, so always unmask.
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaPickPlaceOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the dataset action format.

    The model outputs 32-dim padded actions; we take only the first 7 dims
    (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
