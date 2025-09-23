#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 平均絶対誤差（|native - ref|）の棒グラフ

import argparse, os, math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Meiryo'

CHUNK_LOG2, CHUNK_SIZE, N_CHUNKS = 24, 1<<24, 256
DTYPE, BLOCK = "<f4", 1<<20

def cell_LR_f32(x32):
    """float32 の隣接値で量子化セル [L,R] を作る（RN の中点近似）"""
    x32 = x32.astype(np.float32, copy=False)
    with np.errstate(over='ignore', invalid='ignore'):
        prev32 = np.nextafter(x32, np.float32(-np.inf))
        next32 = np.nextafter(x32, np.float32(np.inf))
    x64 = x32.astype(np.float64, copy=False)
    L = 0.5 * (prev32.astype(np.float64) + x64)
    R = 0.5 * (x64 + next32.astype(np.float64))
    return L, R

def ref_sin_cos_f32(x32):
    """参照: float64 で sin/cos を計算→float32 丸め"""
    x64 = x32.astype(np.float64, copy=False)
    return np.sin(x64).astype(np.float32), np.cos(x64).astype(np.float32)

def scan_and_accumulate(root, xmin, xmax, bins):
    # 等分割エッジ（敷き詰め用）
    edges = np.linspace(xmin, xmax, bins+1, dtype=np.float64)
    bin_w   = np.zeros(bins, dtype=np.float64)  # Σ 重み（セル長）
    bin_s   = np.zeros(bins, dtype=np.float64)  # Σ (abs_err_sin * 重み)
    bin_c   = np.zeros(bins, dtype=np.float64)  # Σ (abs_err_cos * 重み)
    bw = edges[1] - edges[0]
    MIN_NORMAL = np.float32(np.ldexp(1.0, -126))  # 非正規除外

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
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = base_idx if cur == BLOCK else np.arange(cur, dtype=np.uint32)
            idx = (off + idx_local).astype(np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | idx
            x = u.view(np.float32)

            finite = np.isfinite(x)
            non_denorm = (x == np.float32(0.0)) | (np.abs(x) >= MIN_NORMAL)
            m = finite & non_denorm
            if not np.any(m):
                off += cur; continue

            x = x[m]
            ys = np.asarray(ms[off:off+cur][m], np.float32)
            yc = np.asarray(mc[off:off+cur][m], np.float32)
            rs, rc = ref_sin_cos_f32(x)

            # ★ 絶対誤差（float64 で差を取り安全に）
            err_s = np.abs(ys.astype(np.float64) - rs.astype(np.float64))
            err_c = np.abs(yc.astype(np.float64) - rc.astype(np.float64))

            # 量子化セルと範囲の交差長で重み付け
            L, R = cell_LR_f32(x)
            L = np.maximum(L, xmin); R = np.minimum(R, xmax)
            w = R - L
            k = w > 0
            if not np.any(k):
                off += cur; continue
            L, R, w = L[k], R[k], w[k]
            err_s, err_c = err_s[k], err_c[k]

            # 左端が属するビン（右端は xmax で止める）
            b0 = np.floor((L - xmin) / bw).astype(np.int64)
            b0 = np.clip(b0, 0, bins-1)
            cut = edges[b0 + 1]
            same = (R <= cut) | (b0 == bins-1)

            if np.any(same):
                i = same
                np.add.at(bin_w, b0[i], w[i])
                np.add.at(bin_s, b0[i], err_s[i] * w[i])
                np.add.at(bin_c, b0[i], err_c[i] * w[i])

            cross = ~same
            if np.any(cross):
                i = cross; idx0 = b0[i]; idx1 = np.minimum(idx0 + 1, bins - 1)
                w1 = np.maximum(0.0, cut[i] - L[i])
                w2 = np.maximum(0.0, R[i] - cut[i])
                np.add.at(bin_w, idx0, w1); np.add.at(bin_s, idx0, err_s[i] * w1); np.add.at(bin_c, idx0, err_c[i] * w1)
                np.add.at(bin_w, idx1, w2); np.add.at(bin_s, idx1, err_s[i] * w2); np.add.at(bin_c, idx1, err_c[i] * w2)

            off += cur

        del ms, mc
        if (chi % 8) == 0:
            print(f"[progress] chunk {chi}/255")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_abs_s = bin_s / bin_w
        mean_abs_c = bin_c / bin_w

    denom = float(bin_w.sum())
    tot_abs_s = (bin_s.sum() / denom) if denom > 0 else np.nan
    tot_abs_c = (bin_c.sum() / denom) if denom > 0 else np.nan
    return edges, mean_abs_s, mean_abs_c, (tot_abs_s, tot_abs_c)

def plot_bars_filled(edges, y, title, out_png):
    left, width = edges[:-1], np.diff(edges)
    y_plot = y.copy()
    y_plot[~np.isfinite(y_plot)] = np.nan  # NaN は非表示

    plt.figure()
    plt.bar(left, y_plot, width=width, align="edge", linewidth=0, color="#EB9800")  # 敷き詰め・線なし
    # 縦軸はリニア（通常スケール）。自動スケールに任せる。
    plt.xlim(edges[0], edges[-1])
    plt.xlabel("x (引数)")
    plt.ylabel("平均絶対誤差 |native - ref|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Per-x-bin mean absolute error (bars, filled) for native_sin/native_cos")
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

    edges, mean_abs_s, mean_abs_c, (tot_s, tot_c) = scan_and_accumulate(root, xmin, xmax, a.bins)
    print("[result] 範囲全体の平均絶対誤差:")
    print(f"         native_sin: {tot_s:.9e}")
    print(f"         native_cos: {tot_c:.9e}")

    tag = f"_{a.tag}" if a.tag else ""
    plot_bars_filled(edges, mean_abs_s,
                     f"平均絶対誤差 (native_sin) in [{edges[0]}, {edges[-1]}]",
                     os.path.join(root, f"bars_native_sin_abs{tag}.png"))
    plot_bars_filled(edges, mean_abs_c,
                     f"平均絶対誤差 (native_cos) in [{edges[0]}, {edges[-1]}]",
                     os.path.join(root, f"bars_native_cos_abs{tag}.png"))

if __name__ == "__main__":
    main()
