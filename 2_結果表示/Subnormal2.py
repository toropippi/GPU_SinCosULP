#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Subnormal inputs only: per-bin mean ULP bars (native_sin / native_cos)

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Meiryo'

CHUNK_LOG2, CHUNK_SIZE = 24, 1 << 24
DTYPE, BLOCK = "<f4", 1 << 20
MIN_NORMAL_F32 = np.float32(np.ldexp(1.0, -126))  # 1.17549435e-38
# サブノーマルは exp==0 & mantissa!=0（±側：チャンク 0 と 0x80）

def ulp_distance_f32(a32: np.ndarray, b32: np.ndarray) -> np.ndarray:
    """Bruce Dawson 法の ULP 距離（float64 で返す）"""
    ai = a32.view(np.int32).astype(np.int64, copy=False)
    bi = b32.view(np.int32).astype(np.int64, copy=False)
    ai_ord = np.where(ai >= 0, ai, 0x80000000 - ai)
    bi_ord = np.where(bi >= 0, bi, 0x80000000 - bi)
    return np.abs(ai_ord - bi_ord).astype(np.float64)

def scan_bins_subnormal(root: str, func_name: str, bins: int, ref_root: str):
    """
    サブノーマルの x のみを対象に、[-MIN_NORMAL, +MIN_NORMAL] を bins 等分。
    各ビンの ULP 平均（単純平均）を返す。
    """
    xmin = -float(MIN_NORMAL_F32)
    xmax = +float(MIN_NORMAL_F32)
    edges = np.linspace(xmin, xmax, bins + 1, dtype=np.float64)
    bw = edges[1] - edges[0]

    bin_sum = np.zeros(bins, dtype=np.float64)
    bin_cnt = np.zeros(bins, dtype=np.int64)

    target_chunks = (0, 0x80)
    kind = "sin" if func_name.endswith("sin") else "cos"

    for chi in target_chunks:
        p_native = os.path.join(root,     f"{chi}_{func_name}")
        p_ref    = os.path.join(ref_root, f"{chi}_{kind}")
        if not (os.path.exists(p_native) and os.path.exists(p_ref)):
            print(f"[warn] missing files for chunk {chi}: {p_native} / {p_ref}")
            continue

        mm_n = np.memmap(p_native, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mm_r = np.memmap(p_ref,    dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = np.arange(cur, dtype=np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | (off + idx_local)

            # サブノーマル抽出: exp==0 & mantissa!=0
            is_sub = ((u & np.uint32(0x7f800000)) == 0) & ((u & np.uint32(0x007fffff)) != 0)
            if not np.any(is_sub):
                off += cur
                continue

            u_sub = u[is_sub]
            x = u_sub.view(np.float32)
            y_native = np.asarray(mm_n[off:off+cur][is_sub], dtype=np.float32)
            y_ref    = np.asarray(mm_r[off:off+cur][is_sub], dtype=np.float32)

            ulp_err = ulp_distance_f32(y_native, y_ref)

            # x をビンへマッピング（右端は最後のビンに吸収）
            b = np.floor((x.astype(np.float64) - xmin) / bw).astype(np.int64)
            b = np.clip(b, 0, bins - 1)

            # 各ビンに加算
            np.add.at(bin_sum, b, ulp_err)
            np.add.at(bin_cnt, b, 1)

            off += cur

        del mm_n, mm_r

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_ulp = bin_sum / bin_cnt
    return edges, mean_ulp, (bin_cnt.sum(), np.nanmean(mean_ulp))

def plot_bars_filled(edges, y, title, out_png, color="skyblue"):
    left, width = edges[:-1], np.diff(edges)
    mask = np.isfinite(y)  # count=0 のビンは NaN → 非表示
    plt.figure(figsize=(9, 4.8))
    plt.bar(left[mask], y[mask], width=width[mask], align="edge", linewidth=0, color=color)
    plt.xlim(edges[0], edges[-1])
    plt.xlabel("x (subnormal inputs)")
    plt.ylabel("Mean ULP per bin (linear)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Subnormal inputs: per-bin mean ULP bars for native_sin/native_cos")
    ap.add_argument("--dir", required=True, help="Directory containing <chunk>_native_sin / <chunk>_native_cos")
    ap.add_argument("--bins", type=int, default=400, help="Number of x-bins (default: 400)")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    ref_root = os.path.abspath("doubleSample")
    print(f"[info] dir={root}")
    print(f"[info] ref={ref_root} (doubleSample)")
    print(f"[info] bins={args.bins}, range=[{-float(MIN_NORMAL_F32)}, {float(MIN_NORMAL_F32)}]")

    # native_sin
    edges, mean_ulp_sin, (count_sin, mean_of_means_sin) = scan_bins_subnormal(root, "native_sin", args.bins, ref_root)
    print(f"[sin ] subnormals counted: {count_sin:,}, mean( per-bin mean ULP ) = {mean_of_means_sin:.6f}")
    plot_bars_filled(edges, mean_ulp_sin,
                     f"Subnormal inputs: Mean ULP per bin (native_sin, bins={args.bins})",
                     os.path.join(root, f"subnormal_native_sin_ulp_bins.png"),
                     color="skyblue")

    # native_cos
    edges, mean_ulp_cos, (count_cos, mean_of_means_cos) = scan_bins_subnormal(root, "native_cos", args.bins, ref_root)
    print(f"[cos ] subnormals counted: {count_cos:,}, mean( per-bin mean ULP ) = {mean_of_means_cos:.6f}")
    plot_bars_filled(edges, mean_ulp_cos,
                     f"Subnormal inputs: Mean ULP per bin (native_cos, bins={args.bins})",
                     os.path.join(root, f"subnormal_native_cos_ulp_bins.png"),
                     color="skyblue")

if __name__ == "__main__":
    main()
