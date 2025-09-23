#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 平均ULP棒グラフ

import argparse, os, math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Meiryo'

CHUNK_LOG2, CHUNK_SIZE, N_CHUNKS = 24, 1<<24, 256
DTYPE, BLOCK = "<f4", 1<<20

def ulp_distance_f32(a32, b32):
    ai = a32.view(np.int32).astype(np.int64, copy=False)
    bi = b32.view(np.int32).astype(np.int64, copy=False)
    ai = np.where(ai >= 0, ai, 0x80000000 - ai)
    bi = np.where(bi >= 0, bi, 0x80000000 - bi)
    return np.abs(ai - bi)

def cell_LR_f32(x32):
    x32 = x32.astype(np.float32, copy=False)
    prev32 = np.nextafter(x32, np.float32(-np.inf))
    next32 = np.nextafter(x32, np.float32(np.inf))
    x64    = x32.astype(np.float64)
    L = 0.5 * (prev32.astype(np.float64) + x64)
    R = 0.5 * (x64 + next32.astype(np.float64))
    return L, R

def ref_sin_cos_f32(x32):
    x64 = x32.astype(np.float64, copy=False)
    return np.sin(x64).astype(np.float32), np.cos(x64).astype(np.float32)

def scan_and_accumulate(root, xmin, xmax, bins):
    edges   = np.linspace(xmin, xmax, bins+1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    bin_w   = np.zeros(bins, dtype=np.float64)
    bin_s   = np.zeros(bins, dtype=np.float64)
    bin_c   = np.zeros(bins, dtype=np.float64)
    bw = edges[1]-edges[0]
    MIN_NORMAL = np.float32(np.ldexp(1.0, -126))

    base_idx = np.arange(BLOCK, dtype=np.uint32)

    for chi in range(N_CHUNKS):
        ps = os.path.join(root, f"{chi}_native_sin")
        pc = os.path.join(root, f"{chi}_native_cos")
        if not (os.path.exists(ps) and os.path.exists(pc)):
            print(f"[warn] missing chunk {chi}")
            continue
        ms = np.memmap(ps, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mc = np.memmap(pc, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE-off)
            idx_local = base_idx if cur==BLOCK else np.arange(cur, dtype=np.uint32)
            idx = (off + idx_local).astype(np.uint32)
            u = (np.uint32(chi)<<np.uint32(CHUNK_LOG2)) | idx
            x = u.view(np.float32)

            finite = np.isfinite(x)
            non_denorm = (x == np.float32(0.0)) | (np.abs(x) >= MIN_NORMAL)
            m = finite & non_denorm
            if not np.any(m): off += cur; continue

            x = x[m]
            ys = np.asarray(ms[off:off+cur][m], np.float32)
            yc = np.asarray(mc[off:off+cur][m], np.float32)
            rs, rc = ref_sin_cos_f32(x)
            ulps = ulp_distance_f32(ys, rs).astype(np.float64)
            ulpc = ulp_distance_f32(yc, rc).astype(np.float64)

            L, R = cell_LR_f32(x)
            L = np.maximum(L, xmin); R = np.minimum(R, xmax)
            w = R - L
            k = w > 0
            if not np.any(k): off += cur; continue
            L, R, w, ulps, ulpc = L[k], R[k], w[k], ulps[k], ulpc[k]

            b0 = np.floor((L - xmin)/bw).astype(np.int64)
            b0 = np.clip(b0, 0, bins-1)
            cut = edges[b0+1]
            same = (R <= cut) | (b0 == bins-1)

            if np.any(same):
                i = same
                np.add.at(bin_w, b0[i], w[i])
                np.add.at(bin_s, b0[i], ulps[i]*w[i])
                np.add.at(bin_c, b0[i], ulpc[i]*w[i])

            cross = ~same
            if np.any(cross):
                i = cross; idx0 = b0[i]; idx1 = np.minimum(idx0+1, bins-1)
                w1 = np.maximum(0.0, cut[i]-L[i]); w2 = np.maximum(0.0, R[i]-cut[i])
                np.add.at(bin_w, idx0, w1); np.add.at(bin_s, idx0, ulps[i]*w1); np.add.at(bin_c, idx0, ulpc[i]*w1)
                np.add.at(bin_w, idx1, w2); np.add.at(bin_s, idx1, ulps[i]*w2); np.add.at(bin_c, idx1, ulpc[i]*w2)

            off += cur

        del ms, mc
        if (chi % 8) == 0: print(f"[progress] chunk {chi}/255")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_s = bin_s / bin_w
        mean_c = bin_c / bin_w

    denom = float(bin_w.sum())
    tot_s = (bin_s.sum()/denom) if denom>0 else np.nan
    tot_c = (bin_c.sum()/denom) if denom>0 else np.nan
    return centers, edges, mean_s, mean_c, (tot_s, tot_c)

def plot_bars_filled(edges, y, title, out_png):
    left  = edges[:-1]
    width = np.diff(edges)

    y_plot = y.copy()
    y_plot[~np.isfinite(y_plot)] = np.nan
    mask = np.isfinite(y_plot) & (y_plot > 0)

    import matplotlib
    plt.figure()
    plt.bar(left[mask], y_plot[mask], width=width[mask],
            align="edge", linewidth=0)
    plt.yscale("log")
    plt.ylim(10, 1e25)   # ★ 追加：縦軸の最小/最大を固定
    plt.xlim(edges[0], edges[-1])
    plt.xlabel("x (引数)")
    plt.ylabel("重み付け平均 ULP (対数スケール)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Per-x-bin weighted-mean ULP (bars, filled) for native_sin/native_cos")
    ap.add_argument("--dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pi-multiple", type=float)
    g.add_argument("--xmin", type=float)
    ap.add_argument("--xmax", type=float)
    ap.add_argument("--bins", type=int, required=True)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()

    if a.pi_multiple is not None:
        k = float(a.pi_multiple); xmin, xmax = -k*math.pi, +k*math.pi
    else:
        if a.xmax is None: raise SystemExit("--xmax is required when using --xmin")
        xmin, xmax = float(a.xmin), float(a.xmax)
    if not (xmax > xmin): raise SystemExit("Require xmax > xmin")

    root = os.path.abspath(a.dir)
    print(f"[info] dir={root}  range=[{xmin}, {xmax}]  bins={a.bins}")

    centers, edges, mean_s, mean_c, (tot_s, tot_c) = scan_and_accumulate(root, xmin, xmax, a.bins)
    print("[result] 重み付け平均 ULP over the whole range:")
    print(f"         native_sin: {tot_s:.9f} ULP")
    print(f"         native_cos: {tot_c:.9f} ULP")

    tag = f"_{a.tag}" if a.tag else ""
    plot_bars_filled(edges, mean_s,
                     f"重み付け平均 ULP (native_sin) in [-5000pi, 5000pi]",
                     os.path.join(root, f"bars_native_sin_ulp{tag}.png"))
    plot_bars_filled(edges, mean_c,
                     f"重み付け平均 ULP (native_cos) in [-5000pi, 5000pi]",
                     os.path.join(root, f"bars_native_cos_ulp{tag}.png"))

if __name__ == "__main__":
    main()
