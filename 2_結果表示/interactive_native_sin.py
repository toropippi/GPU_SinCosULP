#!/usr/bin/env python3
"""Interactive explorer for GPU native_sin datasets."""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

CHUNK_BITS = 24
CHUNK_SIZE = 1 << CHUNK_BITS
DTYPE = "<f4"


class NativeSinReader:
    """Lazy reader for chunked native_sin outputs."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._cache: dict[int, np.memmap] = {}

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
            chunk = self._load_chunk(int(chunk_id))
            result[mask] = chunk[offsets[mask]]
        return result


class NativeSinExplorer:
    """Interactive viewer that pans and zooms across native_sin(x)."""

    def __init__(self, args: argparse.Namespace) -> None:
        dataset_root = os.path.abspath(args.dir)
        self.reader = NativeSinReader(dataset_root)
        self.samples = max(3, args.samples)
        spacing = float(max(np.spacing(np.float32(args.center)), np.finfo(np.float32).eps))
        self.min_width = max(float(args.min_width), spacing)
        self.max_width = max(float(args.max_width), self.min_width * 1.0001)
        self.center = float(args.center)
        self.width = float(np.clip(args.width, self.min_width, self.max_width))
        self.home_target = float(args.pi_multiple * math.pi) if args.pi_multiple is not None else None
        self.highlight_x = self._unique_sequence(args.highlight_x)
        self._init_plot(args.title)
        self.update_plot()

    @staticmethod
    def _unique_sequence(values: Sequence[float] | None) -> list[float]:
        if not values:
            return []
        seen: set[float] = set()
        unique: list[float] = []
        for value in values:
            v = float(value)
            if v in seen:
                continue
            seen.add(v)
            unique.append(v)
        return unique

    def _init_plot(self, title: str) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        (self.line,) = self.ax.plot([], [], linewidth=1.4, color="#0071C5", label="native_sin(x)")
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlabel("x (float32)")
        self.ax.set_ylabel("native_sin(x)")
        self.ax.set_title(title)
        self.ax.grid(True, linewidth=0.3, alpha=0.25)
        self.ax.margins(x=0)
        self.status = self.ax.text(0.02, 0.02, "", transform=self.ax.transAxes, fontsize=10, ha="left", va="bottom")
        self.instructions = self.fig.text(
            0.5,
            0.96,
            "Scroll: zoom  |  Left/Right: pan  |  Shift slows  |  Alt/Ctrl accelerates  |  Home: jump to primary",
            ha="center",
            va="baseline",
            fontsize=9,
        )
        self.highlight_lines: list[plt.Line2D] = []
        for idx, marker in enumerate(self.highlight_x):
            label = "marker_x" + (" (primary)" if idx == 0 else "") + f" = {marker:g}"
            line = self.ax.axvline(marker, color="#FF8800", linestyle="--", linewidth=1.1, label=label)
            self.highlight_lines.append(line)
        if self.highlight_lines:
            self.ax.legend(loc="upper right")
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _sample_window(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.width / 2.0
        span = np.linspace(self.center - half, self.center + half, self.samples, dtype=np.float64)
        xs32 = span.astype(np.float32)
        xs32 = np.unique(xs32)
        if xs32.size < 2:
            center32 = np.float32(self.center)
            lower = np.nextafter(center32, np.float32(-np.inf), dtype=np.float32)
            upper = np.nextafter(center32, np.float32(np.inf), dtype=np.float32)
            xs32 = np.array([lower, center32, upper], dtype=np.float32)
        xs_plot = xs32.astype(np.float64)
        ys = self.reader.fetch(xs32).astype(np.float64)
        return xs_plot, ys

    def update_plot(self) -> None:
        xs, ys = self._sample_window()
        self.line.set_data(xs, ys)
        self.ax.set_xlim(xs[0], xs[-1])
        for line, marker in zip(self.highlight_lines, self.highlight_x):
            line.set_xdata([marker, marker])
        status_lines = [
            f"center={self.center:.9g}  center/pi={self.center / math.pi:.6f}",
            f"width={self.width:.9g}  samples={self.samples}  unique={xs.size}",
        ]
        if self.home_target is not None:
            status_lines.append(f"home_target={self.home_target:.9g} (k*pi={self.home_target / math.pi:.0f})")
            status_lines.append(f"offset_home={self.center - self.home_target:.6g}")
        if self.highlight_x:
            status_lines.append("markers=" + ", ".join(f"{marker:g}" for marker in self.highlight_x))
        self.status.set_text("\n".join(status_lines))
        self.fig.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if event.step == 0:
            return
        pivot = float(event.xdata) if event.xdata is not None else self.center
        scale = 0.8 if event.step > 0 else 1.25
        new_width = float(np.clip(self.width * scale, self.min_width, self.max_width))
        if math.isclose(new_width, self.width, rel_tol=1e-12, abs_tol=0.0):
            return
        rel = (self.center - pivot) / self.width if self.width else 0.0
        self.center = pivot + rel * new_width
        self.width = new_width
        self.update_plot()

    def on_key(self, event) -> None:
        key = event.key or ""
        if key == "home":
            if self.home_target is not None:
                self.center = self.home_target
                self.update_plot()
            return
        if "left" in key:
            direction = -1.0
        elif "right" in key:
            direction = 1.0
        else:
            return
        modifiers = key.split("+")[:-1]
        step_factor = 0.2
        if "shift" in modifiers:
            step_factor *= 0.2
        if "alt" in modifiers or "ctrl" in modifiers:
            step_factor *= 5.0
        delta = direction * self.width * step_factor
        self.center += delta
        self.update_plot()

    def show(self) -> None:
        self.fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        plt.show()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively inspect native_sin(x) near large arguments.")
    parser.add_argument("--dir", default=os.path.join(".", "intelUHD770_win"),
                        help="Directory containing *_native_sin chunks (default: ./intelUHD770_win)")
    parser.add_argument("--pi-multiple", type=float, default=32768.0,
                        help="Primary k*pi focus used when pressing Home (default: 32768)")
    parser.add_argument("--highlight-x", type=float, nargs="*", default=(114514.0,),
                        help="Add vertical marker lines at these raw x values (default: 114514.0)")
    parser.add_argument("--center", type=float, default=32768.0 * math.pi,
                        help="Initial x center value")
    parser.add_argument("--width", type=float, default=200.0,
                        help="Initial window width in x units")
    parser.add_argument("--min-width", type=float, default=0.01,
                        help="Smallest allowed window width")
    parser.add_argument("--max-width", type=float, default=1e7,
                        help="Largest allowed window width")
    parser.add_argument("--samples", type=int, default=2048,
                        help="Number of x samples drawn across the window")
    parser.add_argument("--title", default="Intel native_sin(x) explorer",
                        help="Window title override")
    return parser.parse_args(argv)


def main() -> None:
    explorer = NativeSinExplorer(parse_args())
    explorer.show()


if __name__ == "__main__":
    main()
