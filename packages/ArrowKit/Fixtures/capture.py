"""Capture ARROW parity fixtures from the reference PyTorch model (CPU, float32).

Run from a checkout of github.com/ifradlin/arrow (6fca6f8) after `uv sync`:

    uv run python /path/to/arrow-swift/packages/ArrowKit/Fixtures/capture.py

Writes beside this script:

- `encode.safetensors`: prepared images, patches, ID tokens and DA3's reference frame,
  for three frames of the repository's railway.mp4 at 224 px (reference 0);
- `reordered.safetensors`: the same for four frames at 112 px, where DA3 picks frame 3,
  so the frames are reordered and restored;
- `decode.safetensors`: a mixed query set on the first clip and every prediction;
- `pose.safetensors`: the pose task's queries, predictions and fitted cameras.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from arrow import ARROW
from arrow.infer import load_input
from arrow.models import encoder as encoder_module
from arrow.tasks.pose import estimate_poses, pose_queries
from arrow.types import Queries

HERE = Path(__file__).parent
CLIP = dict(source="assets/videos/railway.mp4", stride=10, max_frames=3, max_side=224)
REORDERED = dict(source="assets/videos/railway.mp4", stride=10, max_frames=4, max_side=112)

torch.manual_seed(0)
model = ARROW.from_pretrained("ifradlin/arrow", device="cpu")

reference = {}
original_select = encoder_module._select_reference_view


def select(x):
    index = original_select(x)
    reference["index"] = index
    return index


encoder_module._select_reference_view = select


def capture_encoding(clip, name):
    images = load_input(clip["source"], stride=clip["stride"], max_frames=clip["max_frames"], max_side=clip["max_side"]).images
    with torch.inference_mode():
        encoded = model.encode(images)
    case = dict(clip, frames=images.shape[1], height=images.shape[3], width=images.shape[4])
    save_file(
        {
            "images": images.contiguous(),
            "patches": encoded.context_tokens.contiguous(),
            "id_tokens": encoded.id_tokens.contiguous(),
            "reference": reference["index"].to(torch.int64).contiguous(),
        },
        HERE / f"{name}.safetensors",
        metadata={"case": json.dumps(case)},
    )
    return images, encoded, case


capture_encoding(REORDERED, "reordered")
images, encoded, case = capture_encoding(CLIP, "encode")

# A mixed query set: grid pixels, fractional seeds and the image corners, asked across
# source / target / reference roles.
frames, height, width = images.shape[1], images.shape[3], images.shape[4]
generator = torch.Generator().manual_seed(0)
count = 48
xy = torch.stack(
    [torch.rand(count, generator=generator) * width - 0.5, torch.rand(count, generator=generator) * height - 0.5], -1
)
xy[:4] = torch.tensor([[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]])
xy[4:12] = xy[4:12].round()
roles = torch.randint(0, frames, (3, count), generator=generator)
queries = Queries(xy, roles[0], roles[1], roles[2])
with torch.inference_mode():
    predictions = model.decode(encoded, queries)
    context = model.context_proj(encoded.context_tokens)
    tokens = model.query_embed(encoded, queries.batched(), torch.float32)
save_file(
    {
        "xy": queries.xy.contiguous(),
        "source_frame": queries.source_frame.contiguous(),
        "target_frame": queries.target_frame.contiguous(),
        "reference_frame": queries.reference_frame.contiguous(),
        "context": context.contiguous(),
        "tokens": tokens.contiguous(),
        **{f"predictions.{k}": getattr(predictions, k).contiguous() for k in
           ("points", "point_confidence", "uv", "visibility_logits", "normals", "displacement")},
    },
    HERE / "decode.safetensors",
    metadata={"case": json.dumps(case)},
)

# The pose task: a 16 x 16 grid per frame, in its own camera and the middle frame's.
pose = pose_queries(images, grid_size=16, reference_frame="middle")
with torch.inference_mode():
    pose_predictions = model.decode(encoded, pose)
poses = estimate_poses(pose_predictions, pose.xy, num_frames=frames, reference_frame="middle")
save_file(
    {
        "xy": pose.xy.contiguous(),
        "source_frame": pose.source_frame.contiguous(),
        "target_frame": pose.target_frame.contiguous(),
        "reference_frame": pose.reference_frame.contiguous(),
        **{f"predictions.{k}": getattr(pose_predictions, k).contiguous() for k in
           ("points", "point_confidence", "uv", "visibility_logits", "normals", "displacement")},
        "camera_to_reference": poses.camera_to_reference.contiguous(),
        "valid": poses.valid.contiguous(),
        "intrinsics": poses.intrinsics.contiguous(),
    },
    HERE / "pose.safetensors",
    metadata={"case": json.dumps(dict(case, grid_size=16, reference_frame=frames // 2))},
)
print("captured", case, "reference", reference["index"].tolist())
