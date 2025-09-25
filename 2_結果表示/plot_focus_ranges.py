#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フォーカス範囲の y(x) を 3ベンダー + 参照でプロットしてPNG保存。
- native_sin: [0, 0.02],  [pi ± eps]
- native_cos: [pi/2 ± eps]
可視化改善:
- pi/2 と pi の中央に縦基準線
- 凡例のベンダー名は *_win を削除
- 線は細め
- 色: NVIDIA=緑系, AMD=赤系, Intel=青系, 参照=グレー

使い方例:
  python plot_focus_ranges.py --root . --out-dir plots_focus \
    --points 2001 --cos90_eps 0.02 --sinpi_eps 0.02
"""

from __future__ import annotations
import argparse, os, sys, math
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

U24_MASK = (1 << 24) - 1
DEFAULT_GPUS = ["gfx1036_win", "intelUHD770_win", "RTX5090_win"]

# ---- bitcast helpers ----
def f32_to_u32(x: np.float32) -> int:
    return int(np.frombuffer(np.float32(x).tobytes(), dtype="<u4")[0])

def u32_to_f32(u: int) -> np.float32:
    return np.frombuffer(np.uint32(u).tobytes(), dtype="<f4")[0]

# ---- memmap cache ----
class ChunkCache:
    def __init__(self, root: str, dtype: str = "<f4"):
        self.root = root
        self.dtype = dtype
        self._cache: Dict[Tuple[str, str, int], np.memmap] = {}

    def get(self, gpu: str, op: str, chunk: int) -> np.memmap:
        key = (gpu, op, chunk)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.root, gpu, f"{chunk}_{op}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing file: {path}")
        mm = np.memmap(path, dtype=self.dtype, mode="r", shape=(1<<24,))
        self._cache[key] = mm
        return mm

    def get_ref(self, op: str, chunk: int) -> np.memmap:
        key = ("doubleSample", op, chunk)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.root, "doubleSample", f"{chunk}_{op}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing file: {path}")
        mm = np.memmap(path, dtype=self.dtype, mode="r", shape=(1<<24,))
        self._cache[key] = mm
        return mm

# ---- vendor label & color helpers ----
def pretty_label(gpu_dir: str) -> str:
    # 表示名は *_win を外して短く
    name = gpu_dir.replace("_win", "")
    return name

def vendor_color(gpu_dir: str) -> str:
    """
    会社イメージ色:
      - NVIDIA: 緑 (#76B900 相当)
      - AMD: 赤   (#D9232E など)
      - Intel: 青 (#0071C5)
    見つからなければ自動色。
    """
    s = gpu_dir.lower()
    if "rtx" in s or "nvidia" in s:
        return "#76B900"
    if "gfx" in s or "amd" in s:
        return "#D9232E"
    if "intel" in s or "uhd" in s or "arc" in s:
        return "#0071C5"
    return None  # matplotlib に任せる

# ---- data retrieval ----
def load_y_for_xs(xs_f32: np.ndarray, op: str, gpus: List[str], cache: ChunkCache,
                  ref_mode: str = "auto") -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    assert op in ("native_sin", "native_cos")
    N = xs_f32.size
    u32s = np.frombuffer(xs_f32.astype("<f4").tobytes(), dtype="<u4")
    chunks = (u32s >> 24) & 0xFF
    indices = (u32s & U24_MASK)

    vendor_ys: Dict[str, np.ndarray] = {}
    for gpu in gpus:
        ys = np.empty(N, dtype=np.float32)
        for ch in np.unique(chunks):
            sel = np.where(chunks == ch)[0]
            arr = cache.get(gpu, op, int(ch))
            ys[sel] = arr[indices[sel]]
        vendor_ys[gpu] = ys

    # reference
    y_ref = np.empty(N, dtype=np.float32)
    use_doubleSample = False
    if ref_mode in ("auto", "doubleSample"):
        try:
            _ = cache.get_ref("sin" if op == "native_sin" else "cos", int(chunks[0]))
            use_doubleSample = True
        except Exception:
            use_doubleSample = False

    if ref_mode == "doubleSample" and not use_doubleSample:
        raise FileNotFoundError("doubleSample が見つかりません（--ref=doubleSample）。")

    if use_doubleSample:
        ref_op = "sin" if op == "native_sin" else "cos"
        for ch in np.unique(chunks):
            sel = np.where(chunks == ch)[0]
            arr = cache.get_ref(ref_op, int(ch))
            y_ref[sel] = arr[indices[sel]]
    else:
        fun = np.sin if op == "native_sin" else np.cos
        y_ref = fun(xs_f32.astype(np.float64)).astype(np.float32)

    return vendor_ys, y_ref

# ---- plotting ----
def plot_one(xs: np.ndarray,
             vendor_ys: Dict[str, np.ndarray],
             y_ref: np.ndarray,
             title: str,
             out_path: str,
             vline_x: float | None = None,
             xlabel: str = "x (radians)",
             ylabel: str = "y",
             ref_color: str = "#666666",
             ref_lw: float = 1.4,
             vendor_lw: float = 1.0):
    plt.figure(figsize=(9, 5))

    # 正解（やや細め〜標準の間）
    plt.plot(xs, y_ref, label="reference", linewidth=ref_lw, color=ref_color)

    # 各ベンダー（細め）
    for gpu_dir, ys in vendor_ys.items():
        clr = vendor_color(gpu_dir)
        lbl = pretty_label(gpu_dir)
        if clr is None:
            #plt.plot(xs, ys, label=lbl, linewidth=vendor_lw)
            plt.scatter(xs, ys,label=lbl, color=clr, linewidth=vendor_lw, s=1.5)
        else:
            #plt.plot(xs, ys, label=lbl, linewidth=vendor_lw, color=clr)
            plt.scatter(xs, ys,label=lbl,  color=clr, linewidth=vendor_lw, s=1.5)

    # 基準縦線（中心 x）
    if vline_x is not None:
        ymin, ymax = plt.ylim()  # 先に計算（線でスケールが動かないように）
        plt.axvline(vline_x, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
        # 軸範囲を元に戻す
        plt.ylim(ymin, ymax)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---- main ----
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default=".", help="GPU/ doubleSample データのルート")
    ap.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS, help="ベンダー（ディレクトリ名）")
    ap.add_argument("--out-dir", default="plots_focus", help="PNG出力先")
    ap.add_argument("--points", type=int, default=1001, help="各レンジのサンプル数")
    ap.add_argument("--ref", choices=["auto", "doubleSample", "calc"], default="auto",
                    help="参照: auto(優先doubleSample)/doubleSample/calc")
    # 範囲
    ap.add_argument("--sin0_min", type=float, default=0.0, help="native_sin near 0 の最小")
    ap.add_argument("--sin0_max", type=float, default=0.02, help="native_sin near 0 の最大")
    ap.add_argument("--cos90_eps", type=float, default=0.01, help="native_cos near pi/2 の±幅")
    ap.add_argument("--sinpi_eps", type=float, default=0.01, help="native_sin near pi の±幅")
    # 線の太さ（細めが既定）
    ap.add_argument("--vendor_lw", type=float, default=1.0, help="ベンダー線の太さ")
    ap.add_argument("--ref_lw", type=float, default=1.4, help="参照線の太さ")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = ChunkCache(args.root)

    # 1) sin near 0（基準線は不要）
    xs0 = np.linspace(args.sin0_min, args.sin0_max, args.points, dtype=np.float64).astype(np.float32)
    v0, r0 = load_y_for_xs(xs0, "native_sin", args.gpus, cache, ref_mode=args.ref)
    plot_one(xs0.astype(np.float64), v0, r0,
             title="native_sin near 0",
             out_path=os.path.join(args.out_dir, "focus_sin_near0.png"),
             vline_x=None,
             vendor_lw=args.vendor_lw, ref_lw=args.ref_lw)

    # 2) cos near pi/2（中心に縦線）
    pi_over_2 = float(math.pi/2.0)
    xs90 = np.linspace(pi_over_2 - args.cos90_eps, pi_over_2 + args.cos90_eps,
                       args.points, dtype=np.float64).astype(np.float32)
    v90, r90 = load_y_for_xs(xs90, "native_cos", args.gpus, cache, ref_mode=args.ref)
    plot_one(xs90.astype(np.float64), v90, r90,
             title="native_cos near π/2",
             out_path=os.path.join(args.out_dir, "focus_cos_near_pi_over_2.png"),
             vline_x=pi_over_2,
             vendor_lw=args.vendor_lw, ref_lw=args.ref_lw)

    # 3) sin near pi（中心に縦線）
    piv = float(math.pi)
    xs_pi = np.linspace(piv - args.sinpi_eps, piv + args.sinpi_eps,
                        args.points, dtype=np.float64).astype(np.float32)
    vpi, rpi = load_y_for_xs(xs_pi, "native_sin", args.gpus, cache, ref_mode=args.ref)
    plot_one(xs_pi.astype(np.float64), vpi, rpi,
             title="native_sin near π",
             out_path=os.path.join(args.out_dir, "focus_sin_near_pi.png"),
             vline_x=piv,
             vendor_lw=args.vendor_lw, ref_lw=args.ref_lw)

    print("[DONE] saved:")
    for fn in ("focus_sin_near0.png", "focus_cos_near_pi_over_2.png", "focus_sin_near_pi.png"):
        print("  -", os.path.join(args.out_dir, fn))

if __name__ == "__main__":
    main()
