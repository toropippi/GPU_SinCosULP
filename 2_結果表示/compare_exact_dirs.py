#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
検証1:同一ベンダーで差があるか？
2つの結果ディレクトリを比較し、0..255 の各 <chunk>_sin / _cos / _native_sin / _native_cos が
float32 ビット列まで完全一致かを検証。差があればそのファイルごとに [DIFF] を出し、
最初の1件の詳細（位置・入力値・ビット）を表示。全一致なら [OK]。

使い方:
  python compare_exact_dirs.py <dirA> <dirB> [--block 4194304]

戻り値:
  0: 全ファイル完全一致
  1: いずれかで不一致
"""

import os
import sys
import argparse
import numpy as np

CHUNK_LOG2 = 24
CHUNK_SIZE = 1 << CHUNK_LOG2
DTYPE = "<f4"
KINDS = ("sin", "cos", "native_sin", "native_cos")

def open_chunk(root: str, chunk: int, kind: str) -> np.memmap:
    path = os.path.join(root, f"{chunk}_{kind}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # サイズ検査（期待 = 2^24 * 4 bytes）
    expected_bytes = CHUNK_SIZE * 4
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(f"size mismatch: {path} ({actual_bytes} != {expected_bytes})")
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

def bits_u32_from_f32(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint32)

def u32_to_input(chunk: int, idx: int) -> int:
    return ((chunk & 0xFF) << CHUNK_LOG2) | (idx & (CHUNK_SIZE - 1))

def f32_from_u32(u: int) -> np.float32:
    return np.array([u], dtype=np.uint32).view(np.float32)[0]

def hexbits_from_f32(x: np.float32) -> str:
    return f"0x{np.array([x], dtype=np.float32).view(np.uint32)[0]:08x}"

def compare_file_pair(dirA: str, dirB: str, kind: str, chunk: int, block: int):
    """
    1ファイル（同 kind & chunk）の完全一致検証。
    戻り値: (is_equal: bool, mismatch_count: int, first_detail: dict|None)
    """
    mmA = open_chunk(dirA, chunk, kind)
    mmB = open_chunk(dirB, chunk, kind)

    total_diff = 0
    first = None

    for start in range(0, CHUNK_SIZE, block):
        end = min(start + block, CHUNK_SIZE)
        a = np.asarray(mmA[start:end], dtype=np.float32)
        b = np.asarray(mmB[start:end], dtype=np.float32)

        au = bits_u32_from_f32(a)
        bu = bits_u32_from_f32(b)
        diff_mask = (au != bu)

        cnt = int(diff_mask.sum())
        if cnt:
            total_diff += cnt
            if first is None:
                off = int(np.nonzero(diff_mask)[0][0])
                idx = start + off
                av = np.float32(a[off]); bv = np.float32(b[off])
                u  = u32_to_input(chunk, idx)
                x  = f32_from_u32(u)
                first = {
                    "idx": idx,
                    "u": u,
                    "x": float(x),
                    "A": float(av),
                    "B": float(bv),
                    "Abits": hexbits_from_f32(av),
                    "Bbits": hexbits_from_f32(bv),
                }

    del mmA; del mmB
    return (total_diff == 0), total_diff, first

def main():
    ap = argparse.ArgumentParser(description="Exact float32 bitwise comparison across two result directories.")
    ap.add_argument("dirA", help="ディレクトリA（基準）")
    ap.add_argument("dirB", help="ディレクトリB（比較）")
    ap.add_argument("--block", type=int, default=1<<22, help="処理ブロック要素数（既定: 4,194,304）")
    args = ap.parse_args()

    dirA = os.path.abspath(args.dirA)
    dirB = os.path.abspath(args.dirB)
    block = int(args.block)

    print(f"[info] A={dirA}")
    print(f"[info] B={dirB}")
    print(f"[info] CHUNK_SIZE={CHUNK_SIZE:,}, block={block:,}")

    any_diff = False
    files_ok = 0
    files_ng = 0
    missing = 0

    for kind in KINDS:
        for chunk in range(256):
            pathA = os.path.join(dirA, f"{chunk}_{kind}")
            pathB = os.path.join(dirB, f"{chunk}_{kind}")
            if not os.path.exists(pathA) or not os.path.exists(pathB):
                print(f"[MISS] {chunk}_{kind}: missing "
                      f"{'(A)' if not os.path.exists(pathA) else ''}"
                      f"{'(B)' if not os.path.exists(pathB) else ''}")
                any_diff = True
                missing += 1
                continue

            try:
                ok, count, first = compare_file_pair(dirA, dirB, kind, chunk, block)
            except Exception as e:
                print(f"[ERR ] {chunk}_{kind}: {e}")
                any_diff = True
                files_ng += 1
                continue

            if ok:
                print(f"[OK  ] {chunk}_{kind}: all {CHUNK_SIZE:,} values identical")
                files_ok += 1
            else:
                any_diff = True
                files_ng += 1
                if first is not None:
                    print(f"[DIFF] {chunk}_{kind}: mismatches={count:,} "
                          f"(first idx={first['idx']}, u=0x{first['u']:08x}, x={first['x']:.9g}, "
                          f"A={first['A']!r} ({first['Abits']}), "
                          f"B={first['B']!r} ({first['Bbits']}))")
                else:
                    print(f"[DIFF] {chunk}_{kind}: mismatches={count:,}")

    total_files = 256 * len(KINDS)
    print(f"\n[summary] files: total={total_files}, ok={files_ok}, diff={files_ng}, missing={missing}")
    sys.exit(0 if not any_diff else 1)

if __name__ == "__main__":
    main()
