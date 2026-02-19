#!/usr/bin/env python3
"""Fast v3.0 → v2.1 conversion using ffmpeg for video splitting.

Instead of decoding/re-encoding every frame through Python (hours),
this script:
  1. Splits parquets with pandas (seconds)
  2. Splits videos with ffmpeg segment muxer — single decode pass per
     camera, frame-accurate cuts at episode boundaries (~2 min total)
  3. Writes v2.1 metadata jsonl files (instant)

Usage (from the openpi repo root):
  uv run scripts/convert_v3_to_v2.py \
      --src-repo-id DorianAtSchool/pick_place_combined \
      --dst-repo-id DorianAtSchool/pick_place_combined_v2
"""

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_CHUNK_SIZE = 1000


# ── helpers ──────────────────────────────────────────────────────────────────

def load_v3_tasks(src_root: Path) -> dict[int, str]:
    """Load tasks from v3 parquet (schema varies)."""
    tasks_parquet = src_root / "meta" / "tasks.parquet"
    if not tasks_parquet.exists():
        # Try jsonl fallback
        tasks_jsonl = src_root / "meta" / "tasks.jsonl"
        if tasks_jsonl.exists():
            with jsonlines.open(tasks_jsonl) as r:
                return {item["task_index"]: item["task"] for item in r}
        return {0: "pick and place"}

    df = pd.read_parquet(tasks_parquet)
    if "task" in df.columns and "task_index" in df.columns:
        return dict(zip(df["task_index"].astype(int), df["task"].astype(str)))
    # v3 format: task string is the DataFrame index, task_index is a column
    return {int(row["task_index"]): str(idx) for idx, row in df.iterrows()}


def serialize_stats(vals: np.ndarray) -> dict:
    """Compute mean/std/min/max/count along axis 0 and return JSON-serializable dict."""
    return {
        "mean": vals.mean(axis=0).tolist(),
        "std": vals.std(axis=0).tolist(),
        "min": vals.min(axis=0).tolist(),
        "max": vals.max(axis=0).tolist(),
        "count": [len(vals)],
    }


def split_video_with_ffmpeg(
    src_video: Path,
    dst_dir: Path,
    ep_boundaries: list[tuple],
    fps: int,
) -> None:
    """Split a concatenated video into per-episode clips using per-episode ffmpeg calls.

    Uses -ss before -i for fast keyframe-based seeking, then -frames:v for exact
    frame count. Each call takes <1s for a ~140-frame episode.
    """
    for ep_idx, start_g, end_g, n_frames, _ in ep_boundaries:
        start_time = start_g / fps
        dst_video = dst_dir / f"episode_{ep_idx:06d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.6f}",
            "-i", str(src_video),
            "-frames:v", str(n_frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-an", "-loglevel", "error",
            str(dst_video),
        ]
        subprocess.run(cmd, check=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fast convert LeRobot v3 dataset to v2.1 for openpi"
    )
    parser.add_argument("--src-repo-id", required=True)
    parser.add_argument("--dst-repo-id", required=True)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    src_root = HF_LEROBOT_HOME / args.src_repo_id
    dst_root = HF_LEROBOT_HOME / args.dst_repo_id

    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")

    # Clean destination
    if dst_root.exists():
        log.info(f"Removing existing {dst_root}")
        shutil.rmtree(dst_root)

    # ── 1. Read v3 metadata ──────────────────────────────────────────────────
    log.info(f"Reading v3 dataset from {src_root}")
    v3_info = json.loads((src_root / "meta" / "info.json").read_text())
    fps = v3_info["fps"]
    total_episodes = v3_info["total_episodes"]
    total_frames = v3_info["total_frames"]
    features = v3_info["features"]

    tasks = load_v3_tasks(src_root)
    log.info(f"  {total_episodes} episodes, {total_frames} frames, {fps} fps, {len(tasks)} tasks")

    video_keys = sorted(k for k, v in features.items() if v.get("dtype") == "video")
    scalar_keys = [k for k in ["action", "observation.state"] if k in features]
    log.info(f"  Videos: {video_keys}  Scalars: {scalar_keys}")

    # ── 2. Read v3 parquet ───────────────────────────────────────────────────
    data_files = sorted(src_root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files in {src_root / 'data'}")
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    df = df.sort_values("index").reset_index(drop=True)

    # ── 3. Compute episode boundaries ────────────────────────────────────────
    ep_boundaries = []  # list of (ep_idx, start_global_idx, end_global_idx, n_frames, task_idx)
    for ep_idx in range(total_episodes):
        ep_mask = df["episode_index"] == ep_idx
        ep_df = df[ep_mask]
        if len(ep_df) == 0:
            continue
        start_idx = int(ep_df["index"].min())
        end_idx = int(ep_df["index"].max())
        task_idx = int(ep_df["task_index"].iloc[0])
        ep_boundaries.append((ep_idx, start_idx, end_idx, len(ep_df), task_idx))

    # ── 4. Create v2.1 directory structure ───────────────────────────────────
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    (dst_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for vkey in video_keys:
        (dst_root / "videos" / "chunk-000" / vkey).mkdir(parents=True, exist_ok=True)

    # ── 5. Split parquets (per-episode) ──────────────────────────────────────
    log.info("Splitting parquets...")
    global_offset = 0
    episodes_meta = []
    episodes_stats = []
    all_scalar_values = {k: [] for k in scalar_keys}

    for ep_idx, start_g, end_g, n_frames, task_idx in tqdm(ep_boundaries, desc="Parquets"):
        ep_df = df[df["episode_index"] == ep_idx].sort_values("index").copy()

        # Re-index for v2.1: frame_index 0..N-1 within episode, global index continuous
        ep_df["frame_index"] = list(range(n_frames))
        ep_df["index"] = list(range(global_offset, global_offset + n_frames))

        # Drop any v3-only columns that v2.1 doesn't expect
        keep_cols = scalar_keys + ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        keep_cols = [c for c in keep_cols if c in ep_df.columns]
        ep_df = ep_df[keep_cols]

        chunk = ep_idx // DEFAULT_CHUNK_SIZE
        chunk_dir = dst_root / "data" / f"chunk-{chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(chunk_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        # Episode metadata
        task_str = tasks.get(task_idx, "pick and place")
        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [task_str],
            "length": n_frames,
        })

        # Episode stats
        ep_stats = {}
        for key in scalar_keys:
            if key in ep_df.columns:
                vals = np.stack(ep_df[key].values).astype(np.float32)
                ep_stats[key] = serialize_stats(vals)
                all_scalar_values[key].append(vals)
        episodes_stats.append({"episode_index": ep_idx, "stats": ep_stats})

        global_offset += n_frames

    # ── 6. Split videos with ffmpeg ──────────────────────────────────────────
    log.info("Splitting videos with ffmpeg (per-episode)...")
    for vkey in tqdm(video_keys, desc="Cameras"):
        src_video = src_root / "videos" / vkey / "chunk-000" / "file-000.mp4"
        if not src_video.exists():
            log.warning(f"  Video not found: {src_video}, skipping")
            continue
        dst_dir = dst_root / "videos" / "chunk-000" / vkey
        split_video_with_ffmpeg(src_video, dst_dir, ep_boundaries, fps)

    # Verify video counts
    for vkey in video_keys:
        dst_dir = dst_root / "videos" / "chunk-000" / vkey
        n_mp4s = len(list(dst_dir.glob("*.mp4")))
        if n_mp4s != total_episodes:
            log.warning(f"  {vkey}: expected {total_episodes} videos, got {n_mp4s}")

    # ── 7. Write v2.1 metadata ───────────────────────────────────────────────
    log.info("Writing v2.1 metadata...")

    # info.json
    v2_info = {
        "codebase_version": "v2.1",
        "robot_type": v3_info.get("robot_type", "franka"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": (total_episodes - 1) // DEFAULT_CHUNK_SIZE + 1,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video_keys else None,
        "features": features,  # keep v3 features (schema compatible)
    }
    with open(dst_root / "meta" / "info.json", "w") as f:
        json.dump(v2_info, f, indent=4)

    # tasks.jsonl
    with jsonlines.open(dst_root / "meta" / "tasks.jsonl", "w") as w:
        for task_idx in sorted(tasks):
            w.write({"task_index": task_idx, "task": tasks[task_idx]})

    # episodes.jsonl
    with jsonlines.open(dst_root / "meta" / "episodes.jsonl", "w") as w:
        w.write_all(episodes_meta)

    # episodes_stats.jsonl
    with jsonlines.open(dst_root / "meta" / "episodes_stats.jsonl", "w") as w:
        w.write_all(episodes_stats)

    # stats.json (aggregate)
    agg_stats = {}
    for key in scalar_keys:
        if all_scalar_values[key]:
            vals = np.concatenate(all_scalar_values[key], axis=0)
            agg_stats[key] = serialize_stats(vals)
    with open(dst_root / "meta" / "stats.json", "w") as f:
        json.dump(agg_stats, f, indent=4)

    log.info(f"✓ v2.1 dataset ready at {dst_root}")
    log.info(f"  {total_episodes} episodes, {total_frames} frames, {len(video_keys)} cameras")

    # ── 8. Optionally push to hub ────────────────────────────────────────────
    if args.push_to_hub:
        log.info("Pushing to HuggingFace Hub...")
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(args.dst_repo_id)
        ds.push_to_hub(
            tags=["franka", "pick_place", "openpi"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        log.info("Push complete!")


if __name__ == "__main__":
    main()
