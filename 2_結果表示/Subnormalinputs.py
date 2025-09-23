#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Subnormal inputs only: plot native_sin (from --dir) vs reference sin (from ./doubleSample), y = symlog

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Meiryo'

CHUNK_LOG2, CHUNK_SIZE = 24, 1 << 24
DTYPE, BLOCK = "<f4", 1 << 20
SUBNORMALS_PER_SIGN = (1 << 23) - 1          # 8,388,607
TOTAL_SUBNORMALS    = SUBNORMALS_PER_SIGN*2  # 16,777,214
PLOT_MAX_POINTS     = 300_000                # 描画点の最大（計算は全点で実施）

def scan_subnormal_native_vs_ref(root: str, ref_root: str):
    """
    サブノーマル x（exp==0 & mantissa!=0）だけを走査し、
    (x, y_native, y_ref) を等間引きして返す（x 昇順）。
    """
    target_chunks = (0, 0x80)  # サブノーマルが出るチャンク（+/-）
    stride = max(1, TOTAL_SUBNORMALS // PLOT_MAX_POINTS)

    xs_all, yn_all, yr_all = [], [], []
    seen = 0  # サブノーマル入力を何個見たか（グローバルカウンタ）

    for chi in target_chunks:
        p_native = os.path.join(root,     f"{chi}_native_sin")
        p_ref    = os.path.join(ref_root, f"{chi}_sin")
        if not os.path.exists(p_native):
            print(f"[warn] missing file: {p_native}")
            continue
        if not os.path.exists(p_ref):
            print(f"[warn] missing REF file: {p_ref}")
            continue

        mm_n = np.memmap(p_native, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mm_r = np.memmap(p_ref,    dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = np.arange(cur, dtype=np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | (off + idx_local)

            # サブノーマル: exp==0 & mantissa!=0
            is_sub = ((u & np.uint32(0x7f800000)) == 0) & ((u & np.uint32(0x007fffff)) != 0)
            if not np.any(is_sub):
                off += cur
                continue

            u_sub = u[is_sub]
            x = u_sub.view(np.float32)
            y_native = np.asarray(mm_n[off:off+cur][is_sub], dtype=np.float32)
            y_ref    = np.asarray(mm_r[off:off+cur][is_sub], dtype=np.float32)

            # グローバル等間引き（seen を基準）
            local_idx = np.arange(x.size, dtype=np.int64)
            pick = ((seen + local_idx) % stride) == 0

            xs_all.append(x[pick].astype(np.float64))
            yn_all.append(y_native[pick].astype(np.float64))
            yr_all.append(y_ref[pick].astype(np.float64))

            seen += x.size
            off  += cur

        del mm_n, mm_r

    if not xs_all:
        return np.array([]), np.array([]), np.array([])

    x = np.concatenate(xs_all)
    yn = np.concatenate(yn_all)
    yr = np.concatenate(yr_all)

    order = np.argsort(x, kind="mergesort")
    return x[order], yn[order], yr[order]

def plot_native_vs_ref(x, y_native, y_ref, out_png):
    plt.figure(figsize=(9, 4.8))
    # symlog: 負値も描ける対数スケール（ゼロ近傍はリニア）
    plt.yscale('symlog', linthresh=1e-40, linscale=1.0)
    plt.plot(x, y_native, lw=1.0, color="skyblue",  label="native_sin")
    plt.plot(x, y_ref,    lw=1.0, color="#EB9800",  label="ref sin (doubleSample)")
    plt.xlabel("x (subnormal inputs)")
    plt.ylabel("function value (symlog scale)")
    plt.title("Subnormal inputs: native_sin vs reference sin (from doubleSample)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[saved] {out_png}")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot native_sin vs reference sin (doubleSample) for subnormal inputs")
    ap.add_argument("--dir", required=True, help="Directory containing <chunk>_native_sin")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    ref_root = os.path.abspath("doubleSample")  # ★ 正解は ./doubleSample 固定（従来どおり引数は --dir のみ）
    print(f"[info] dir={root}")
    print(f"[info] ref={ref_root} (doubleSample)")

    x, yn, yr = scan_subnormal_native_vs_ref(root, ref_root)
    if x.size == 0:
        print("[warn] no subnormal points found or files missing")
        return

    out_png = os.path.join(root, "subnormal_native_sin_vs_ref.png")
    plot_native_vs_ref(x, yn, yr, out_png)

if __name__ == "__main__":
    main()
