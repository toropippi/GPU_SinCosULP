#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare native_sin/native_cos against doubleSample reference and report
max absolute error (and its x) over [-2π, 2π] and [-100π, 100π].

Usage:
  python max_abs_error_native_vs_double.py --dir gfx1036_win
  # ref dir is ./doubleSample by default; change via --ref-dir
"""

import argparse, os, math, sys
import numpy as np

CHUNK_LOG2, CHUNK_SIZE, N_CHUNKS = 24, 1 << 24, 256
DTYPE, BLOCK = "<f4", 1 << 20
MIN_NORMAL = np.float32(np.ldexp(1.0, -126))  # 非正規の閾値（ゼロは許容）

def bits_hex_f32(x32: np.ndarray) -> str:
    return f"0x{np.asarray(x32, dtype=np.float32).view(np.uint32).item():08x}"

def fmt_pow2(x: float) -> str:
    if not np.isfinite(x) or x <= 0.0:
        return "n/a"
    p = math.log(x, 2.0)
    return f"≈ 2^{p:.2f}"

def update_max(err: np.ndarray, x: np.ndarray, rec: dict):
    """err と x（同長）の中で最大誤差とそのときの x を rec に反映。"""
    if err.size == 0:
        return
    j = int(np.argmax(err))
    m = float(err[j])
    if m > rec["max"]:
        rec["max"] = m
        rec["x"] = float(np.float32(x[j]))
        rec["x_bits"] = bits_hex_f32(np.float32(x[j]))

def run_scan(test_dir: str, ref_dir: str):
    ranges = {
        "±2π":   (-2.0 * math.pi,  +2.0 * math.pi),
        "±100π": (-100.0 * math.pi, +100.0 * math.pi),
    }
    res = {
        "sin": {k: {"max": -1.0, "x": None, "x_bits": None} for k in ranges},
        "cos": {k: {"max": -1.0, "x": None, "x_bits": None} for k in ranges},
    }

    base_idx = np.arange(BLOCK, dtype=np.uint32)

    for chi in range(N_CHUNKS):
        p_native_s = os.path.join(test_dir, f"{chi}_native_sin")
        p_native_c = os.path.join(test_dir, f"{chi}_native_cos")
        p_ref_s    = os.path.join(ref_dir,  f"{chi}_sin")
        p_ref_c    = os.path.join(ref_dir,  f"{chi}_cos")

        if not (os.path.exists(p_native_s) and os.path.exists(p_native_c)):
            # 欠けているチャンクはスキップ
            # print(f"[warn] missing native chunk {chi}", file=sys.stderr)
            continue
        if not (os.path.exists(p_ref_s) and os.path.exists(p_ref_c)):
            print(f"[warn] missing reference chunk {chi}", file=sys.stderr)
            continue

        ms = np.memmap(p_native_s, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mc = np.memmap(p_native_c, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        rs = np.memmap(p_ref_s,    dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        rc = np.memmap(p_ref_c,    dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = base_idx if cur == BLOCK else np.arange(cur, dtype=np.uint32)
            idx = (off + idx_local).astype(np.uint32)

            # 入力 x（u32 ビット列→f32）
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | idx
            x = u.view(np.float32)

            # フィルタ: 有限 かつ （ゼロはOK or 非正規でない）
            finite = np.isfinite(x)
            non_denorm = (x == np.float32(0.0)) | (np.abs(x) >= MIN_NORMAL)
            msk = finite & non_denorm
            if not np.any(msk):
                off += cur
                continue

            x = x[msk]
            ys = np.asarray(ms[off:off+cur][msk], dtype=np.float32)
            yc = np.asarray(mc[off:off+cur][msk], dtype=np.float32)
            yr_s = np.asarray(rs[off:off+cur][msk], dtype=np.float32)  # doubleSample の正解（sin）
            yr_c = np.asarray(rc[off:off+cur][msk], dtype=np.float32)  # doubleSample の正解（cos）

            # 絶対誤差（float64 で安全に）
            err_s = np.abs(ys.astype(np.float64) - yr_s.astype(np.float64))
            err_c = np.abs(yc.astype(np.float64) - yr_c.astype(np.float64))

            x64 = x.astype(np.float64, copy=False)
            for key, (xmin, xmax) in ranges.items():
                r = (x64 >= xmin) & (x64 <= xmax)
                if np.any(r):
                    update_max(err_s[r], x[r], res["sin"][key])
                    update_max(err_c[r], x[r], res["cos"][key])

            off += cur

        del ms, mc, rs, rc
        if (chi % 8) == 0:
            print(f"[progress] chunk {chi}/255", file=sys.stderr)

    return res

def main():
    ap = argparse.ArgumentParser(description="Max abs error of native sin/cos vs doubleSample over two ranges")
    ap.add_argument("--dir", required=True, help="Directory containing <chunk>_native_sin / <chunk>_native_cos")
    ap.add_argument("--ref-dir", default="doubleSample", help="Directory containing reference <chunk>_sin / <chunk>_cos (default: ./doubleSample)")
    args = ap.parse_args()

    test_dir = os.path.abspath(args.dir)
    ref_dir  = os.path.abspath(args.ref_dir)
    print(f"[info] test_dir={test_dir}")
    print(f"[info] ref_dir ={ref_dir}")

    res = run_scan(test_dir, ref_dir)

    def show_one(name: str, entry: dict):
        for key in ("±2π", "±100π"):
            e = entry[key]
            if e["max"] < 0:
                print(f"[{name}] range {key}: no data")
                continue
            print(f"[{name}] range {key}:")
            print(f"  max |error|   : {e['max']:.9e}  {fmt_pow2(e['max'])}")
            print(f"  at x          : {e['x']}  (bits={e['x_bits']})")

    show_one("native_sin", res["sin"])
    show_one("native_cos", res["cos"])

if __name__ == "__main__":
    main()
