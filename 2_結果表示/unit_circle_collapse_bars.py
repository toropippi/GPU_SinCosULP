#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit-circle collapse: 1 - (sin^2 + cos^2) over x.
- Files: prefer <chunk>_native_sin / <chunk>_native_cos (float32, len=2^24). Missing chunks are skipped.
- Range: --pi-multiple K  -> [-K*pi, +K*pi]  (or --xmin/--xmax)
- Metric: --metric {ulp, abs}
- Bars: --bins N (e.g., 40000), filled (no gaps)
- Weighting: per-x float32 quantization cell length intersecting the range (RN cell), as before.

Usage examples:
  python unit_circle_collapse_bars.py --dir RTX5090_win --pi-multiple 1 --metric ulp  --bins 40000
  python unit_circle_collapse_bars.py --dir RTX5090_win --pi-multiple 100 --metric abs --bins 40000
"""

import argparse, os, math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Meiryo'

CHUNK_LOG2, CHUNK_SIZE, N_CHUNKS = 24, 1<<24, 256
DTYPE, BLOCK = "<f4", 1<<20
MIN_NORMAL = np.float32(np.ldexp(1.0, -126))  # 1.17549435e-38

def ulp_distance_f32(a32: np.ndarray, b32: np.ndarray) -> np.ndarray:
    """Bruce Dawson 法で ULP 距離（float64で返す）。"""
    ai = a32.view(np.int32).astype(np.int64, copy=False)
    bi = b32.view(np.int32).astype(np.int64, copy=False)
    ai_ord = np.where(ai >= 0, ai, 0x80000000 - ai)
    bi_ord = np.where(bi >= 0, bi, 0x80000000 - bi)
    return np.abs(ai_ord - bi_ord).astype(np.float64)

def cell_LR_f32(x32: np.ndarray):
    """float32 の隣接値で量子化セル [L,R]（RNの中点近似）。"""
    x32 = x32.astype(np.float32, copy=False)
    with np.errstate(over='ignore', invalid='ignore'):
        prev32 = np.nextafter(x32, np.float32(-np.inf))
        next32 = np.nextafter(x32, np.float32(np.inf))
    x64 = x32.astype(np.float64, copy=False)
    L = 0.5 * (prev32.astype(np.float64) + x64)
    R = 0.5 * (x64 + next32.astype(np.float64))
    return L, R

def choose_paths(root: str, chi: int):
    """優先: native_*。なければ非nativeにフォールバックせず、None を返してスキップ（安定のため）。"""
    pns = os.path.join(root, f"{chi}_native_sin")
    pnc = os.path.join(root, f"{chi}_native_cos")
    if os.path.exists(pns) and os.path.exists(pnc):
        return pns, pnc, "native"
    # 必要なら下のコメントアウトを外せば非nativeにフォールバック可能
    # ps = os.path.join(root, f"{chi}_sin"); pc = os.path.join(root, f"{chi}_cos")
    # if os.path.exists(ps) and os.path.exists(pc):
    #     return ps, pc, "highprec"
    return None, None, None

def scan_and_accumulate(root, xmin, xmax, bins, metric):
    edges = np.linspace(xmin, xmax, bins+1, dtype=np.float64)
    bw = edges[1] - edges[0]
    bin_w  = np.zeros(bins, dtype=np.float64)  # Σ 重み
    bin_e  = np.zeros(bins, dtype=np.float64)  # Σ (誤差 * 重み)
    zeros  = None  # ULP参照用

    base_idx = np.arange(BLOCK, dtype=np.uint32)

    for chi in range(N_CHUNKS):
        p_sin, p_cos, kind = choose_paths(root, chi)
        if not p_sin:
            # print(f"[warn] skip chunk {chi} (files missing)")
            continue
        ms = np.memmap(p_sin, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mc = np.memmap(p_cos, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = base_idx if cur == BLOCK else np.arange(cur, dtype=np.uint32)
            idx = (off + idx_local).astype(np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | idx
            x = u.view(np.float32)

            # 入力フィルタ: 有限かつ非サブノーマル（ゼロは含む）
            finite = np.isfinite(x)
            non_denorm = (x == np.float32(0.0)) | (np.abs(x) >= MIN_NORMAL)
            m = finite & non_denorm
            if not np.any(m):
                off += cur; continue

            x = x[m]
            ys = np.asarray(ms[off:off+cur][m], dtype=np.float32)  # sin(x)
            yc = np.asarray(mc[off:off+cur][m], dtype=np.float32)  # cos(x)

            # e = 1 - (sin^2 + cos^2) を float32 演算で作る（実装寄り）
            s2 = ys * ys                    # float32
            c2 = yc * yc                    # float32
            sumsq = s2 + c2                 # float32
            e = np.float32(1.0) - sumsq     # float32 (崩れ量)

            if metric == "abs":
                err = np.abs(e.astype(np.float64))  # 絶対誤差
            else:  # 'ulp'：0.0f からの ULP 距離
                if zeros is None or zeros.shape[0] != e.shape[0]:
                    zeros = np.zeros_like(e, dtype=np.float32)
                err = ulp_distance_f32(e, zeros)    # float64

            # 量子化セル重み
            L, R = cell_LR_f32(x)
            L = np.maximum(L, xmin); R = np.minimum(R, xmax)
            w = R - L
            k = w > 0
            if not np.any(k):
                off += cur; continue
            L, R, w, err = L[k], R[k], w[k], err[k]

            b0 = np.floor((L - xmin) / bw).astype(np.int64)
            b0 = np.clip(b0, 0, bins-1)
            cut = edges[b0 + 1]
            same = (R <= cut) | (b0 == bins-1)

            if np.any(same):
                i = same
                np.add.at(bin_w, b0[i], w[i])
                np.add.at(bin_e, b0[i], err[i] * w[i])

            cross = ~same
            if np.any(cross):
                i = cross; idx0 = b0[i]; idx1 = np.minimum(idx0 + 1, bins - 1)
                w1 = np.maximum(0.0, cut[i] - L[i])
                w2 = np.maximum(0.0, R[i] - cut[i])
                np.add.at(bin_w, idx0, w1); np.add.at(bin_e, idx0, err[i] * w1)
                np.add.at(bin_w, idx1, w2); np.add.at(bin_e, idx1, err[i] * w2)

            off += cur

        del ms, mc
        if (chi % 8) == 0:
            print(f"[progress] chunk {chi}/255")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_err = bin_e / bin_w
    denom = float(bin_w.sum())
    total_mean = (bin_e.sum() / denom) if denom > 0 else np.nan
    return edges, mean_err, total_mean

def plot_bars(edges, y, metric, out_png):
    left, width = edges[:-1], np.diff(edges)
    mask = np.isfinite(y)
    plt.figure(figsize=(10, 5))
    color = "skyblue" if metric == "ulp" else "#EB9800"
    plt.bar(left[mask], y[mask], width=width[mask], align="edge", linewidth=0, color=color)
    if metric == "ulp":
        plt.yscale("log")  # ULP は対数
        plt.ylabel("Weighted mean ULP of (1 - (sin^2+cos^2))")
    else:
        plt.ylabel("Weighted mean |1 - (sin^2+cos^2)|")
    plt.xlim(edges[0], edges[-1])
    plt.xlabel("x (argument)")
    title_metric = "ULP (log)" if metric == "ulp" else "Absolute (linear)"
    plt.title(f"Unit-circle collapse 1 - (sin^2+cos^2) — {title_metric}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Per-bin bars of 1 - (sin^2+cos^2) over x (native_sin/cos)")
    ap.add_argument("--dir", required=True, help="Folder containing <chunk>_native_sin / <chunk>_native_cos")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pi-multiple", type=float, help="Use range [-k*pi, +k*pi]")
    g.add_argument("--xmin", type=float, help="Explicit xmin (use with --xmax)")
    ap.add_argument("--xmax", type=float, help="Explicit xmax (required if --xmin)")
    ap.add_argument("--metric", choices=["ulp", "abs"], required=True, help="Plot metric: 'ulp' or 'abs'")
    ap.add_argument("--bins", type=int, default=40000, help="Number of equal x-bins (default: 40000)")
    ap.add_argument("--tag", type=str, default="", help="Optional tag for output filenames")
    args = ap.parse_args()

    if args.pi_multiple is not None:
        k = float(args.pi_multiple); xmin, xmax = -k*math.pi, +k*math.pi
    else:
        if args.xmax is None: raise SystemExit("--xmax is required when using --xmin")
        xmin, xmax = float(args.xmin), float(args.xmax)
    if not (xmax > xmin): raise SystemExit("Require xmax > xmin")

    root = os.path.abspath(args.dir)
    print(f"[info] dir={root}")
    print(f"[info] range=[{xmin}, {xmax}]  bins={args.bins}  metric={args.metric}")

    edges, mean_err, tot_mean = scan_and_accumulate(root, xmin, xmax, args.bins, args.metric)
    print(f"[result] Weighted-mean over whole range ({args.metric}): {tot_mean:.9e}")

    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(root, f"unit_circle_collapse_{args.metric}_bins{args.bins}{tag}.png")
    plot_bars(edges, mean_err, args.metric, out)

if __name__ == "__main__":
    main()
