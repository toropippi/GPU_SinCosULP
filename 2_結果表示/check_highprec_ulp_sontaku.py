#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel/AMD の高精度版 sin/cos を doubleSample の正解と比較し、
最大ULPと平均ULPを出力する“忖度版”（±0 を同値扱い）スクリプト。

使い方:
  python check_highprec_ulp_sontaku.py --dir gfx1036_win
  python check_highprec_ulp_sontaku.py --dir gfx1036_win --ref-dir doubleSample
  # 忖度を無効化したい場合:
  python check_highprec_ulp_sontaku.py --dir gfx1036_win --no-ignore-negzero
"""

import argparse, os, sys
import numpy as np

CHUNK_LOG2, CHUNK_SIZE, N_CHUNKS = 24, 1<<24, 256
DTYPE, BLOCK = "<f4", 1<<20

def bits_hex(x32: np.ndarray) -> str:
    return f"0x{np.asarray(x32, dtype=np.float32).view(np.uint32).item():08x}"

def normalize_signed_zero(x: np.ndarray) -> np.ndarray:
    """
    x が +0.0/-0.0 のものは符号ビットをクリアし +0.0 に揃える。
    （それ以外は変更しない）
    """
    x = np.asarray(x, dtype=np.float32)
    xi = x.view(np.uint32)
    zero_mask = (x == np.float32(0.0))  # ±0 とも True
    xi = np.where(zero_mask, xi & np.uint32(0x7fffffff), xi)
    return xi.view(np.float32)

def ulp_distance_f32(a32: np.ndarray, b32: np.ndarray, ignore_negzero: bool = True) -> np.ndarray:
    """
    Bruce Dawson 法の ULP 距離。ignore_negzero=True のとき、±0 の差は ULP=0 とする。
    """
    if ignore_negzero:
        a32 = normalize_signed_zero(a32)
        b32 = normalize_signed_zero(b32)

    ai = a32.view(np.int32).astype(np.int64, copy=False)
    bi = b32.view(np.int32).astype(np.int64, copy=False)
    # 単調空間へ写像
    ai_ord = np.where(ai >= 0, ai, 0x80000000 - ai)
    bi_ord = np.where(bi >= 0, bi, 0x80000000 - bi)
    return np.abs(ai_ord - bi_ord)

def run_scan(test_dir: str, ref_dir: str, ignore_negzero: bool):
    min_normal = np.float32(np.ldexp(1.0, -126))  # 非サブノーマル閾値
    total_cnt = 0

    sum_ulp_s = 0
    sum_ulp_c = 0
    max_ulp_s = -1
    max_ulp_c = -1
    arg_at_max_s = None  # (x32, ulp, y_test, y_ref)
    arg_at_max_c = None

    base_idx = np.arange(BLOCK, dtype=np.uint32)

    for chi in range(N_CHUNKS):
        ts = os.path.join(test_dir, f"{chi}_sin")
        tc = os.path.join(test_dir, f"{chi}_cos")
        rs = os.path.join(ref_dir,  f"{chi}_sin")
        rc = os.path.join(ref_dir,  f"{chi}_cos")

        if not (os.path.exists(ts) and os.path.exists(tc)):
            print(f"[warn] missing test files for chunk {chi}", file=sys.stderr)
            continue
        if not (os.path.exists(rs) and os.path.exists(rc)):
            print(f"[warn] missing REF files for chunk {chi} (dir={ref_dir})", file=sys.stderr)
            continue

        ms = np.memmap(ts, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mc = np.memmap(tc, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        rs_mm = np.memmap(rs, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        rc_mm = np.memmap(rc, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = base_idx if cur == BLOCK else np.arange(cur, dtype=np.uint32)
            idx = (off + idx_local).astype(np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | idx
            x = u.view(np.float32)

            # 有効入力: 有限かつ非サブノーマル（ゼロは含む）
            finite = np.isfinite(x)
            non_denorm = (x == np.float32(0.0)) | (np.abs(x) >= min_normal)
            m = finite & non_denorm
            if not np.any(m):
                off += cur
                continue

            xs = x[m]
            ys = np.asarray(ms[off:off+cur][m], dtype=np.float32, copy=False)
            yc = np.asarray(mc[off:off+cur][m], dtype=np.float32, copy=False)
            rs_block = np.asarray(rs_mm[off:off+cur][m], dtype=np.float32, copy=False)
            rc_block = np.asarray(rc_mm[off:off+cur][m], dtype=np.float32, copy=False)

            ulp_s = ulp_distance_f32(ys, rs_block, ignore_negzero=ignore_negzero)
            ulp_c = ulp_distance_f32(yc, rc_block, ignore_negzero=ignore_negzero)

            sum_ulp_s += ulp_s.sum(dtype=np.int64)
            sum_ulp_c += ulp_c.sum(dtype=np.int64)
            total_cnt += xs.size

            # 最大 ULP の位置（最初に出たもの）
            blk_max_s = int(ulp_s.max(initial=-1))
            blk_max_c = int(ulp_c.max(initial=-1))
            if blk_max_s > max_ulp_s:
                j = int(ulp_s.argmax())
                max_ulp_s = blk_max_s
                arg_at_max_s = (np.float32(xs[j]), blk_max_s, np.float32(ys[j]), np.float32(rs_block[j]))
            if blk_max_c > max_ulp_c:
                j = int(ulp_c.argmax())
                max_ulp_c = blk_max_c
                arg_at_max_c = (np.float32(xs[j]), blk_max_c, np.float32(yc[j]), np.float32(rc_block[j]))

            off += cur

        del ms, mc, rs_mm, rc_mm

        if (chi % 8) == 0:
            print(f"[progress] chunk {chi}/255  counted={total_cnt:,}")

    mean_ulp_s = (sum_ulp_s / total_cnt) if total_cnt > 0 else float("nan")
    mean_ulp_c = (sum_ulp_c / total_cnt) if total_cnt > 0 else float("nan")

    return {
        "count": total_cnt,
        "sin": {
            "max_ulp": max_ulp_s,
            "mean_ulp": mean_ulp_s,
            "arg_at_max": arg_at_max_s,
        },
        "cos": {
            "max_ulp": max_ulp_c,
            "mean_ulp": mean_ulp_c,
            "arg_at_max": arg_at_max_c,
        },
    }

def main():
    ap = argparse.ArgumentParser(description="High-precision sin/cos vs doubleSample ULP (±0 を同値扱いする忖度版)")
    ap.add_argument("--dir", required=True, help="被検体 <chunk>_sin / <chunk>_cos のディレクトリ（例: gfx1036_win）")
    ap.add_argument("--ref-dir", default="doubleSample", help="正解 <chunk>_sin / <chunk>_cos のディレクトリ（既定: doubleSample）")
    ap.add_argument("--ignore-negzero", dest="ignore_negzero", action="store_true", default=True,
                    help="±0 の差は ULP=0 とみなす（既定: 有効）")
    ap.add_argument("--no-ignore-negzero", dest="ignore_negzero", action="store_false",
                    help="±0 も区別して ULP を計算する")
    args = ap.parse_args()

    test_dir = os.path.abspath(args.dir)
    ref_dir  = os.path.abspath(args.ref_dir)

    print(f"[info] test_dir={test_dir}")
    print(f"[info] ref_dir ={ref_dir}")
    print(f"[info] ignore_negzero={args.ignore_negzero}")

    res = run_scan(test_dir, ref_dir, ignore_negzero=args.ignore_negzero)

    def show(name, entry):
        x32, maxulp, y_test, y_ref = entry["arg_at_max"] if entry["arg_at_max"] is not None else (None, None, None, None)
        print(f"\n[{name}]")
        print(f"  count(valid inputs) : {res['count']:,}")
        print(f"  max ULP             : {entry['max_ulp']}")
        if x32 is not None:
            print(f"    at x              : {float(x32)}  (bits={bits_hex(x32)})")
            print(f"    test/ref outputs  : {float(y_test)} / {float(y_ref)}")
        print(f"  mean ULP            : {entry['mean_ulp']:.6f}")
        print(f"  within 2 ULP?       : {'YES' if entry['max_ulp'] is not None and entry['max_ulp'] <= 2 else 'NO'}")

    show("sin", res["sin"])
    show("cos", res["cos"])

if __name__ == "__main__":
    main()
