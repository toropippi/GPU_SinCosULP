#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if high-precision sin(x) returns x bitwise for ALL subnormal float32 inputs.
- 対象: --dir にある <chunk>_sin （chunk=0, 0x80 のみ走査）
- 判定: 出力 y の f32 ビットが入力 x の f32 ビットと一致するか（ビット完全一致）

使い方:
  python check_subnormal_sin_equals_x.py --dir gfx1036_win
"""

import argparse, os, sys
import numpy as np

CHUNK_LOG2, CHUNK_SIZE = 24, 1 << 24
DTYPE, BLOCK = "<f4", 1 << 20  # 内部ブロック（固定でOK）

def run_check(test_dir: str, max_report: int = 10):
    """
    サブノーマル x だけ抽出し、<chunk>_sin の出力と x のビット一致を検査。
    戻り値: (summary_dict, mismatches_list up to max_report)
    """
    target_chunks = (0, 0x80)  # サブノーマルが存在するチャンク
    total_subs = 0
    total_eq   = 0
    mismatches = []

    for chi in target_chunks:
        fpath = os.path.join(test_dir, f"{chi}_sin")
        if not os.path.exists(fpath):
            print(f"[warn] missing file: {fpath}", file=sys.stderr)
            continue

        mm = np.memmap(fpath, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

        off = 0
        while off < CHUNK_SIZE:
            cur = min(BLOCK, CHUNK_SIZE - off)
            idx_local = np.arange(cur, dtype=np.uint32)
            u = (np.uint32(chi) << np.uint32(CHUNK_LOG2)) | (off + idx_local)

            # サブノーマル: exp==0 & mantissa!=0（±0 は除外）
            is_sub = ((u & np.uint32(0x7f800000)) == 0) & ((u & np.uint32(0x007fffff)) != 0)
            if not np.any(is_sub):
                off += cur
                continue

            u_sub = u[is_sub]
            x = u_sub.view(np.float32)  # 入力
            y = np.asarray(mm[off:off+cur][is_sub], dtype=np.float32, copy=False)  # 出力

            bx = x.view(np.uint32)
            by = y.view(np.uint32)
            eq = (bx == by)

            total_subs += eq.size
            total_eq   += int(eq.sum())

            if not np.all(eq) and len(mismatches) < max_report:
                k = np.nonzero(~eq)[0]
                for j in k:
                    mismatches.append({
                        "x": float(np.float32(x[j])),
                        "x_bits": f"0x{int(bx[j]):08x}",
                        "y": float(np.float32(y[j])),
                        "y_bits": f"0x{int(by[j]):08x}",
                    })
                    if len(mismatches) >= max_report:
                        break

            off += cur

        del mm

    all_match = (total_subs > 0) and (total_eq == total_subs)
    return (
        {
            "total_subnormals_checked": total_subs,
            "total_equal": total_eq,
            "ratio_equal": (total_eq / total_subs) if total_subs else float("nan"),
            "all_match": all_match,
        },
        mismatches
    )

def main():
    ap = argparse.ArgumentParser(description="Check if high-precision sin(x) equals x bitwise for subnormal inputs")
    ap.add_argument("--dir", required=True, help="高精度版 <chunk>_sin のディレクトリ（例: gfx1036_win）")
    ap.add_argument("--max-report", type=int, default=10, help="不一致の詳細を最大いくつ出すか（既定:10）")
    args = ap.parse_args()

    test_dir = os.path.abspath(args.dir)
    print(f"[info] test_dir={test_dir}")

    summary, mism = run_check(test_dir, max_report=args.max_report)

    print("\n=== Summary (subnormal inputs, sin) ===")
    print(f"checked        : {summary['total_subnormals_checked']:,}")
    print(f"equal (bitwise): {summary['total_equal']:,}")
    print(f"equal ratio    : {summary['ratio_equal']:.9f}")
    print(f"all match?     : {'YES' if summary['all_match'] else 'NO'}")

    if mism:
        print(f"\n--- First {len(mism)} mismatches ---")
        for i, e in enumerate(mism, 1):
            print(f"[{i}] x={e['x']} (bits={e['x_bits']})  ->  y={e['y']} (bits={e['y_bits']})")

if __name__ == "__main__":
    main()
