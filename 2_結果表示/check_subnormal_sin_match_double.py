#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check whether high-precision sin outputs match doubleSample exactly
for ALL subnormal float32 inputs (bitwise equality).

Usage:
  python check_subnormal_exact_match_sin.py --dir gfx1036_win
  # 正解ディレクトリを変える場合:
  python check_subnormal_exact_match_sin.py --dir gfx1036_win --ref-dir doubleSample
"""

import argparse, os, sys
import numpy as np

CHUNK_LOG2, CHUNK_SIZE = 24, 1 << 24
DTYPE, BLOCK = "<f4", 1 << 20  # 内部ブロックサイズ（ユーザ指定不要）

def bits_hex_f32(x32: np.ndarray) -> str:
    return f"0x{np.asarray(x32, dtype=np.float32).view(np.uint32).item():08x}"

def run_check(test_dir: str, ref_dir: str, max_report: int = 10):
    """
    サブノーマル入力のみ対象。test_dir/<chi>_sin と ref_dir/<chi>_sin を比較。
    返り値: dict(summary), mismatches(list of dict up to max_report)
    """
    target_chunks = (0, 0x80)  # サブノーマルが出現するチャンク（+/-）
    total_subs = 0
    total_eq   = 0
    mismatches = []
    base_idx = np.arange(BLOCK, dtype=np.uint32)

    for chi in target_chunks:
        tpath = os.path.join(test_dir, f"{chi}_sin")
        rpath = os.path.join(ref_dir,  f"{chi}_sin")
        if not (os.path.exists(tpath) and os.path.exists(rpath)):
            print(f"[warn] missing file(s) for chunk {chi}: {tpath} / {rpath}", file=sys.stderr)
            continue

        mt = np.memmap(tpath, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
        mr = np.memmap(rpath, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = base_idx if cur == BLOCK else np.arange(cur, dtype=np.uint32)
            idx = (off + idx_local).astype(np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | idx

            # サブノーマル: exp==0 & mantissa!=0
            is_sub = ((u & np.uint32(0x7f800000)) == 0) & ((u & np.uint32(0x007fffff)) != 0)
            if not np.any(is_sub):
                off += cur
                continue

            u_sub = u[is_sub]
            x = u_sub.view(np.float32)

            yt = np.asarray(mt[off:off+cur][is_sub], dtype=np.float32, copy=False)
            yr = np.asarray(mr[off:off+cur][is_sub], dtype=np.float32, copy=False)

            # ビット一致を判定
            bt = yt.view(np.uint32)
            br = yr.view(np.uint32)
            eq = (bt == br)

            total_subs += eq.size
            total_eq   += int(eq.sum())

            # 不一致があれば最大 max_report 件だけ詳細を取る
            if not np.all(eq) and len(mismatches) < max_report:
                k = np.nonzero(~eq)[0]
                for j in k:
                    mismatches.append({
                        "x":    np.float32(x[j]),
                        "x_bits": f"0x{u_sub[j]:08x}",
                        "test": np.float32(yt[j]),
                        "test_bits": f"0x{bt[j]:08x}",
                        "ref":  np.float32(yr[j]),
                        "ref_bits":  f"0x{br[j]:08x}",
                    })
                    if len(mismatches) >= max_report:
                        break

            off += cur

        del mt, mr

    all_match = (total_subs > 0) and (total_eq == total_subs)
    return (
        {
            "total_subnormals_checked": total_subs,
            "total_equal": total_eq,
            "all_match": all_match
        },
        mismatches
    )

def main():
    ap = argparse.ArgumentParser(description="Check exact match for subnormal inputs: high-precision sin vs doubleSample")
    ap.add_argument("--dir", required=True, help="高精度版 <chunk>_sin のディレクトリ（例: gfx1036_win）")
    ap.add_argument("--ref-dir", default="doubleSample", help="正解 <chunk>_sin のディレクトリ（既定: doubleSample）")
    ap.add_argument("--max-report", type=int, default=10, help="詳細表示する不一致の最大件数（既定: 10）")
    args = ap.parse_args()

    test_dir = os.path.abspath(args.dir)
    ref_dir  = os.path.abspath(args.ref_dir)

    print(f"[info] test_dir={test_dir}")
    print(f"[info] ref_dir ={ref_dir}")
    summary, mism = run_check(test_dir, ref_dir, max_report=args.max_report)

    print("\n=== Summary (subnormal inputs, sin) ===")
    print(f"checked : {summary['total_subnormals_checked']:,}")
    print(f"equal   : {summary['total_equal']:,}")
    print(f"all match? -> {'YES' if summary['all_match'] else 'NO'}")

    if mism:
        print(f"\n--- First {len(mism)} mismatches ---")
        for i, e in enumerate(mism, 1):
            print(f"[{i}] x={float(e['x'])} (bits={e['x_bits']})")
            print(f"    test={float(e['test'])} (bits={e['test_bits']})")
            print(f"    ref ={float(e['ref'])}  (bits={e['ref_bits']})")

if __name__ == "__main__":
    main()
