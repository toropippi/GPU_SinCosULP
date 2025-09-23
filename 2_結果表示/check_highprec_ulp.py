#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD/Intel の高精度版 sin/cos が 2 ULP 以内かを全域で検査。
- 比較対象（被検体）: --dir にある <chunk>_sin / <chunk>_cos （chunk=0..255）
- 正解: --ref-dir (= default: "doubleSample") 内の <chunk>_sin / <chunk>_cos
- 対象引数 x: 有限かつ非サブノーマル（ゼロは含む）

出力:
  * sin / cos それぞれの 最大ULP・平均ULP
  * 最大ULP が出たときの引数 x（float32 値・bit 表示）

使い方例:
  python check_highprec_ulp.py --dir gfx1036_win
  python check_highprec_ulp.py --dir gfx1036_win --ref-dir doubleSample
"""

import argparse, os, sys
import numpy as np

CHUNK_LOG2, CHUNK_SIZE, N_CHUNKS = 24, 1<<24, 256
DTYPE, BLOCK = "<f4", 1<<20  # memmap とブロックサイズ

# ----- ULP 距離（Bruce Dawson 法）-----
def ulp_distance_f32(a32: np.ndarray, b32: np.ndarray) -> np.ndarray:
    ai = a32.view(np.int32).astype(np.int64, copy=False)
    bi = b32.view(np.int32).astype(np.int64, copy=False)
    ai_ord = np.where(ai >= 0, ai, 0x80000000 - ai)
    bi_ord = np.where(bi >= 0, bi, 0x80000000 - bi)
    return np.abs(ai_ord - bi_ord)

def bits_hex(x32: np.ndarray) -> str:
    return f"0x{np.asarray(x32, dtype=np.float32).view(np.uint32).item():08x}"

def run_scan(test_dir: str, ref_dir: str):
    # 統計用アキュムレータ
    min_normal = np.float32(np.ldexp(1.0, -126))  # 非サブノーマル閾値
    total_cnt = 0

    sum_ulp_s = 0
    sum_ulp_c = 0
    max_ulp_s = -1
    max_ulp_c = -1
    arg_at_max_s = None  # (x32, ulp)
    arg_at_max_c = None

    base_idx = np.arange(BLOCK, dtype=np.uint32)

    # 256 チャンクを順に
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
            # 入力 x を u32 のビット列から生成
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | idx
            x = u.view(np.float32)

            # 入力のフィルタ: 有限かつ非サブノーマル（ゼロはOK）
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

            # ULP 計算
            ulp_s = ulp_distance_f32(ys, rs_block)
            ulp_c = ulp_distance_f32(yc, rc_block)

            # 平均用に和を蓄積
            sum_ulp_s += ulp_s.sum(dtype=np.int64)
            sum_ulp_c += ulp_c.sum(dtype=np.int64)
            total_cnt += xs.size

            # 最大更新（最初に現れた x を記録）
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

        # memmap 解放
        del ms, mc, rs_mm, rc_mm

        if (chi % 8) == 0:
            print(f"[progress] chunk {chi}/255  counted={total_cnt:,}")

    # 平均 ULP（単純平均: 有効入力の一様重み）
    mean_ulp_s = (sum_ulp_s / total_cnt) if total_cnt > 0 else float("nan")
    mean_ulp_c = (sum_ulp_c / total_cnt) if total_cnt > 0 else float("nan")

    return {
        "count": total_cnt,
        "sin": {
            "max_ulp": max_ulp_s,
            "mean_ulp": mean_ulp_s,
            "arg_at_max": arg_at_max_s,  # (x32, maxULP, y_test, y_ref)
        },
        "cos": {
            "max_ulp": max_ulp_c,
            "mean_ulp": mean_ulp_c,
            "arg_at_max": arg_at_max_c,
        },
    }

def main():
    ap = argparse.ArgumentParser(description="Check high-precision sin/cos against reference (doubleSample) in ULP.")
    ap.add_argument("--dir", required=True, help="被検体の <chunk>_sin / <chunk>_cos があるディレクトリ（例: gfx1036_win）")
    ap.add_argument("--ref-dir", default="doubleSample", help="正解 <chunk>_sin / <chunk>_cos のディレクトリ（既定: doubleSample）")
    args = ap.parse_args()

    test_dir = os.path.abspath(args.dir)
    ref_dir  = os.path.abspath(args.ref_dir)

    print(f"[info] test_dir={test_dir}")
    print(f"[info] ref_dir ={ref_dir}")
    res = run_scan(test_dir, ref_dir)

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
