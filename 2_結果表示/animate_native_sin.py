#!/usr/bin/env python3
"""Animate native_sin(x) comparisons across GPU vendors."""

import argparse
import math
import os
from collections import deque
from typing import Deque, Dict, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

CHUNK_BITS = 24
CHUNK_SIZE = 1 << CHUNK_BITS
DTYPE = "<f4"

def vendor_color(path: str) -> str | None:
    name = os.path.basename(path).lower()
    if "rtx" in name or "nvidia" in name:
        return "#76B900"  # NVIDIA green
    if "gfx" in name or "amd" in name:
        return "#D9232E"  # AMD red
    if "intel" in name or "uhd" in name or "arc" in name:
        return "#0071C5"  # Intel blue
    return None


class NativeSinReader:
    """Lazy loader for native_sin datasets stored in chunked binary files."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._cache: Dict[int, np.memmap] = {}

    def _load_chunk(self, chunk_id: int) -> np.memmap:
        if chunk_id not in self._cache:
            path = os.path.join(self.root, f"{chunk_id}_native_sin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing native_sin chunk: {path}")
            self._cache[chunk_id] = np.memmap(path, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        return self._cache[chunk_id]

    def fetch(self, xs: Sequence[np.float32]) -> np.ndarray:
        x_arr = np.asarray(xs, dtype=np.float32)
        u_arr = x_arr.view(np.uint32)
        chunks = (u_arr >> CHUNK_BITS).astype(np.int64)
        offsets = (u_arr & ((1 << CHUNK_BITS) - 1)).astype(np.int64)
        result = np.empty_like(x_arr, dtype=np.float32)

        for chunk_id in np.unique(chunks):
            mask = chunks == chunk_id
            mem = self._load_chunk(int(chunk_id))
            result[mask] = mem[offsets[mask]]
        return result


def build_x_sequence(zero_end: float, target_pi: float, frames_zero: int,
                     frames_target_pre: int, frames_target_post: int, transition_target: float,
                     target_window: float) -> tuple[np.ndarray, float, float, float]:
    if frames_zero > 1:
        base_step = zero_end / (frames_zero - 1)
    else:
        base_step = zero_end if zero_end > 0 else 1.0

    segment1 = np.linspace(0.0, zero_end, frames_zero, dtype=np.float64) if frames_zero > 0 else np.empty(0)

    pivot = float(target_pi * math.pi)
    tt = float(np.clip(transition_target, -0.999999, 0.999999))

    if abs(tt) < 1e-9:
        x_transition = pivot
    else:
        delta = float(math.asin(abs(tt)))
        if tt < 0.0:
            x_transition = pivot - delta
        else:
            x_transition = pivot + delta
    delta = abs(pivot - x_transition)

    if frames_target_pre > 0:
        pre_indices = np.arange(frames_target_pre, 0, -1, dtype=np.float64)
        pre = x_transition - base_step * pre_indices
    else:
        pre = np.empty(0)

    if frames_target_post > 0:
        post_indices = np.arange(frames_target_post, dtype=np.float64)
        post = x_transition + base_step * post_indices
    else:
        post = np.empty(0)

    sequence = np.concatenate([segment1, pre, post]).astype(np.float32)
    highlight_window = max(target_window, delta + base_step)
    return sequence, x_transition, highlight_window, base_step


def setup_writer():
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    except ImportError:
        pass


def animate(args: argparse.Namespace) -> None:
    setup_writer()

    vendors = [
        ("Intel", args.intel_dir),
        ("AMD", args.amd_dir),
        ("NVIDIA", args.nvidia_dir),
    ]

    readers = []
    base_rgba = []
    for label, path in vendors:
        abs_path = os.path.abspath(path)
        reader = NativeSinReader(abs_path)
        clr = vendor_color(path) or "#555555"
        readers.append((label, reader, clr))
        base_rgba.append(mcolors.to_rgba(clr, 1.0))

    frames_zero = args.frames_zero
    frames_target_pre = args.frames_target_pre
    frames_target_post = args.frames_target_post

    xs, x_transition, highlight_window, base_step = build_x_sequence(
        args.zero_range,
        args.skip_multiple,
        frames_zero,
        frames_target_pre,
        frames_target_post,
        args.transition_target,
        args.target_window,
    )

    pre_capture_start = x_transition - base_step * args.frames_target_pre
    pre_history_cutoff = x_transition - base_step * (args.frames_target_pre - 1)

    transition_frame = max(frames_zero - 1, 0)
    fade_frames = 36
    intro_title_frames = 45
    trail_length_frames = 40
    trail_history: Deque[np.ndarray] = deque(maxlen=trail_length_frames)
    pre_history: Deque[np.ndarray] = deque(maxlen=trail_length_frames)
    trail_activation_frame: int | None = None
    highlight_triggered = False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(-0.5, len(readers) - 0.5)
    ax.set_ylim(-1.2, 1.6)
    ax.axhline(0.0, color="#999999", linewidth=1, linestyle="--")
    ax.axhline(1.0, color="#cccccc", linewidth=0.5)
    ax.axhline(-1.0, color="#cccccc", linewidth=0.5)
    ax.set_xticks(range(len(readers)))
    ax.set_xticklabels([label for label, *_ in readers])
    ax.tick_params(axis='x', labelsize=26)
    ax.set_ylabel("native_sin(x)")
    ax.set_title("native_sin behaviour across GPU vendors")

    highlight_patch = ax.axvspan(-0.5, len(readers) - 0.5, color="#fff1b8", alpha=0.0, zorder=-2)

    fade_patch = Rectangle(
        (-0.5, -1.2), len(readers), 2.8, transform=ax.transData,
        color="#f4f1ff", alpha=0.0, zorder=4
    )
    ax.add_patch(fade_patch)

    dot_size = 2200.0
    scatter = ax.scatter([], [], s=dot_size, zorder=5)
    trail_scatter = ax.scatter([], [], s=dot_size * 0.7, zorder=3)

    annotations = [ax.text(i, 0.0, "", ha="center", va="center", color="white", fontsize=12,
                           fontweight="bold", zorder=6) for i in range(len(readers))]
    info_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, ha="left", va="top", fontsize=13,
                        fontweight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="#555555"))

    alert_text = ax.text(0.98, 0.94, "", transform=ax.transAxes, ha="right", va="top",
                         fontsize=17, fontweight="bold", color="#D5232E",
                         bbox=dict(facecolor="white", alpha=0.95, edgecolor="#D5232E", boxstyle="round,pad=0.3"))
    intro_text = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center",
                         fontsize=75, fontweight="bold", color="#111111", zorder=7)

    threshold = args.skip_multiple * math.pi
    blink_period = 66

    def near_threshold(x: float) -> bool:
        return abs(x - threshold) <= highlight_window

    def init():
        scatter.set_offsets(np.zeros((len(readers), 2)))
        scatter.set_facecolors(base_rgba)
        trail_scatter.set_offsets(np.empty((0, 2)))
        trail_scatter.set_facecolors([])
        info_text.set_text("")
        alert_text.set_text("")
        intro_text.set_text("")
        intro_text.set_alpha(1.0)
        highlight_patch.set_alpha(0.0)
        fade_patch.set_alpha(0.0)
        trail_history.clear()
        pre_history.clear()
        return [scatter, trail_scatter, *annotations, info_text, alert_text, intro_text, highlight_patch, fade_patch]

    def update(frame_idx: int):
        nonlocal trail_activation_frame, highlight_triggered
        x = xs[frame_idx]
        positions = []
        for i, (_label, reader, _colour) in enumerate(readers):
            y_val = float(reader.fetch([x])[0])
            positions.append([i, y_val])
            annotations[i].set_position((i, y_val))
            annotations[i].set_text(f"{y_val:+.3f}")
        positions_arr = np.array(positions)

        if x >= max(pre_capture_start, pre_history_cutoff - base_step * trail_length_frames):
            pre_history.append(positions_arr.copy())
        else:
            pre_history.clear()

        is_near = near_threshold(x)
        if is_near and trail_activation_frame is None:
            trail_activation_frame = frame_idx
            trail_history.clear()
            for pos in pre_history:
                trail_history.append(pos.copy())

        trail_active = (
            trail_activation_frame is not None
            and frame_idx - trail_activation_frame < trail_length_frames
        )

        if trail_active and trail_history:
            trail_history.append(positions_arr.copy())
            trail_offsets = []
            trail_colors = []
            age = max(frame_idx - trail_activation_frame, 0)
            global_decay = max(0.0, 0.8 - 0.03 * age)
            for idx, pos in enumerate(reversed(trail_history)):
                alpha =  global_decay * math.exp(-0.25 * idx)
                for vendor_idx in range(len(readers)):
                    rgba = list(base_rgba[vendor_idx])
                    rgba[3] = min(max(alpha, 0.0), 1.0)
                    trail_colors.append(rgba)
                    trail_offsets.append([vendor_idx, pos[vendor_idx, 1]])
            trail_scatter.set_offsets(np.array(trail_offsets))
            trail_scatter.set_facecolors(trail_colors)
        else:
            trail_history.clear()
            trail_scatter.set_offsets(np.empty((0, 2)))
            trail_scatter.set_facecolors([])

        scatter.set_offsets(positions_arr)

        info_text.set_text(
            f"x = {float(x):.6f}\n"
            "y = native_sin(x)"
        )

        if frame_idx < intro_title_frames:
            intro_text.set_text("sin(x)")
            intro_text.set_alpha(1.0)
        else:
            intro_text.set_text("")

        if is_near or highlight_triggered:
            highlight_patch.set_alpha(0.4)
            highlight_triggered = True
        else:
            highlight_patch.set_alpha(0.0)

        if frame_idx >= transition_frame:
            progress = (frame_idx - transition_frame) / fade_frames
            if 0.0 <= progress < 1.0:
                fade_patch.set_alpha(0.55 * (1.0 - progress))
            else:
                fade_patch.set_alpha(0.0)
        else:
            fade_patch.set_alpha(0.0)

        if x >= threshold:
            if (frame_idx % blink_period) < (blink_period // 2):
                alert_text.set_color("#000000")  # 黒
                alert_text.set_text(f"x exceeded 32768π")
            else:
                alert_text.set_color("#D5232E")  # 赤
                alert_text.set_text(f"x exceeded 32768π")
                
        else:
            alert_text.set_text("")

        return [scatter, trail_scatter, *annotations, info_text, alert_text, intro_text, highlight_patch, fade_patch]

    anim = FuncAnimation(fig, update, frames=len(xs), init_func=init, blit=True)

    codec = "ffmpeg"
    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".avi":
        writer_kwargs = dict(fps=args.fps, codec="png", bitrate=-1)
    else:
        writer_kwargs = dict(fps=args.fps, codec="libx264", bitrate=args.bitrate)

    anim.save(args.output, writer=codec, dpi=160, savefig_kwargs={"facecolor": "#f9f9f9"}, **writer_kwargs)
    print(f"[saved] {args.output}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vendor comparison video for native_sin(x)")
    parser.add_argument("--intel-dir", default=os.path.join(".", "intelUHD770_win"))
    parser.add_argument("--amd-dir", default=os.path.join(".", "gfx1036_win"))
    parser.add_argument("--nvidia-dir", default=os.path.join(".", "RTX5090_win"))
    parser.add_argument("--output", default="native_sin_vendor_comparison.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bitrate", type=int, default=1800)
    parser.add_argument("--zero-range", type=float, default=9.9467,
                        help="Upper bound for the initial x sweep")
    parser.add_argument("--frames-zero", type=int, default=167,
                        help="Number of frames used in the initial sweep")
    parser.add_argument("--skip-multiple", type=float, default=32768.0,
                        help="Multiplier k for π to jump to (x = kπ)")
    parser.add_argument("--target-window", type=float, default=20.0,
                        help="Half-width around kπ to inspect")
    parser.add_argument("--frames-target-pre", type=int, default=117,
                        help="Frames allotted before hitting kπ within the window")
    parser.add_argument("--frames-target-post", type=int, default=455,
                        help="Frames allotted after exceeding kπ within the window")
    parser.add_argument("--transition-target", type=float, default=-0.5,
                        help="Desired native_sin value at the scene transition")
    return parser.parse_args(argv)


if __name__ == "__main__":
    animate(parse_args())
