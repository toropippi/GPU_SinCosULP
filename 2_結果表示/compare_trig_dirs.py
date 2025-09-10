#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2つのディレクトリを比較して、0..255 の各 <chunk>_sin / <chunk>_cos を
全要素（2^24=16,777,216）ULP比較。NaN同士は一致扱い。
1 ULPでも違えば詳細をprint（--limit 件まで）。
最後に観測した最大ULP差を 1 行で表示。

使い方:
  python compare_trig_dirs.py <dirA> <dirB> [--block 4194304] [--limit 0]
"""

import os
import sys
import argparse
import numpy as np

CHUNK_LOG2 = 24
CHUNK_SIZE = 1 << CHUNK_LOG2
DTYPE = "<f4"  # little-endian float32
KINDS = ("sin", "cos")

# ---------------- helpers ----------------

def bits_u32_from_f32(x: np.ndarray) -> np.ndarray:
    return x.view(np.uint32)

def to_ordered_u32(u: np.ndarray) -> np.ndarray:
    # (u>>31) は 0 or 1。0xFFFFFFFF を掛けることで全ビットマスクに拡張。
    mask = (u >> np.uint32(31)) * np.uint32(0xFFFFFFFF)
    # 正: u ^ 0 ^ 0x80000000 → u | 0x80000000
    # 負: u ^ 0xFFFFFFFF ^ 0x80000000 → (~u) ^ 0x80000000（距離は版Aと同じ）
    return (u ^ mask ^ np.uint32(0x80000000)).astype(np.uint32)

def ulp_distance_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua = bits_u32_from_f32(a)
    ub = bits_u32_from_f32(b)
    oa = to_ordered_u32(ua).astype(np.int64)
    ob = to_ordered_u32(ub).astype(np.int64)
    d = np.abs(oa - ob).astype(np.uint32)
    return d

def u32_to_input(chunk: int, idx: int) -> int:
    return ((chunk & 0xFF) << CHUNK_LOG2) | (idx & (CHUNK_SIZE - 1))

def f32_from_u32(u: int) -> np.float32:
    return np.array([u], dtype=np.uint32).view(np.float32)[0]

def hexbits_from_f32(x: np.float32) -> str:
    return f"0x{np.array([x], dtype=np.float32).view(np.uint32)[0]:08x}"

def open_chunk(path_root: str, chunk: int, kind: str) -> np.memmap:
    p = os.path.join(path_root, f"{chunk}_{kind}")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.memmap(p, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

class MaxTracker:
    def __init__(self) -> None:
        self.max_ulp = 0
        self.info = None  # dict with fields

    def consider(self, kind: str, chunk: int, idx: int, a: np.float32, b: np.float32, ulp: int):
        if ulp > self.max_ulp:
            self.max_ulp = int(ulp)
            u = u32_to_input(chunk, idx)
            x = f32_from_u32(u)
            self.info = {
                "kind": kind,
                "chunk": chunk,
                "idx": idx,
                "u": u,
                "x": float(x),
                "A": float(a),
                "B": float(b),
                "Abits": hexbits_from_f32(np.float32(a)),
                "Bbits": hexbits_from_f32(np.float32(b)),
            }

# ---------------- core ----------------

def process_pair(dirA: str, dirB: str, kind: str, chunk: int, block: int,
                 print_budget: int, tracker: MaxTracker) -> int:
    """
    1ファイル（同一 kind & chunk の A/B）を比較。
    print_budget: 残り印字可能件数。0 は印字禁止、負値は無制限扱い。
    返り値: 実際にprintした件数の増分。
    """
    mmA = open_chunk(dirA, chunk, kind)
    mmB = open_chunk(dirB, chunk, kind)

    printed = 0
    for start in range(0, CHUNK_SIZE, block):
        end = min(start + block, CHUNK_SIZE)
        a = np.asarray(mmA[start:end], dtype=np.float32)
        b = np.asarray(mmB[start:end], dtype=np.float32)

        both_nan = np.isnan(a) & np.isnan(b)
        d = ulp_distance_f32(a, b)
        d[both_nan] = 0

        # 最大ULPの更新（このブロック内）
        # ブロック内の最大を先に見つけて、必要なら詳細を取り出す
        local_max = int(d.max())
        if local_max > 0:
            # すべての不一致を走査しつつ、印字と最大更新を行う
            mism_idxs = np.nonzero(d != 0)[0]
            for off in mism_idxs:
                idx = start + off
                ulp = int(d[off])
                av = np.float32(a[off]); bv = np.float32(b[off])

                # 最大ULPのトラッキング（印字有無に関係なく）
                tracker.consider(kind, chunk, idx, av, bv, ulp)

                # 印字は予算内のみ
                if print_budget != 0 and printed >= max(print_budget, 0):
                    continue

                u = u32_to_input(chunk, idx)
                x = f32_from_u32(u)
                print(
                    f"[{kind}] chunk={chunk:3d} idx={idx:7d}  "
                    f"u=0x{u:08x} x={float(x):.9g}  "
                    f"A={float(av)!r} ({hexbits_from_f32(av)})  "
                    f"B={float(bv)!r} ({hexbits_from_f32(bv)})  "
                    f"ULP={ulp}"
                )
                printed += 1

    del mmA; del mmB
    return printed

def main():
    ap = argparse.ArgumentParser(description="Compare sin/cos results between two directories (ULP-level).")
    ap.add_argument("dirA", help="ディレクトリA（基準）")
    ap.add_argument("dirB", help="ディレクトリB（比較）")
    ap.add_argument("--block", type=int, default=1<<22, help="処理ブロック要素数（既定: 4,194,304）")
    ap.add_argument("--limit", type=int, default=0, help="表示上限（0=無制限）")
    args = ap.parse_args()

    dirA = os.path.abspath(args.dirA)
    dirB = os.path.abspath(args.dirB)
    block = int(args.block)
    limit = int(args.limit)

    print(f"[info] A={dirA}")
    print(f"[info] B={dirB}")
    print(f"[info] CHUNK_SIZE={CHUNK_SIZE:,}, block={block:,}, limit={limit}")

    tracker = MaxTracker()
    total_printed = 0

    for kind in KINDS:
        for chunk in range(256):
            pathA = os.path.join(dirA, f"{chunk}_{kind}")
            pathB = os.path.join(dirB, f"{chunk}_{kind}")
            if not (os.path.exists(pathA) and os.path.exists(pathB)):
                print(f"[warn] missing: {pathA if not os.path.exists(pathA) else ''} {pathB if not os.path.exists(pathB) else ''}")
                continue

            budget = (limit - total_printed) if (limit > 0) else -1  # -1=無制限
            printed = process_pair(dirA, dirB, kind, chunk, block, budget, tracker)
            total_printed += printed

    # --- 最後に最大ULP差を 1 行表示 ---
    if tracker.info is None:
        print("[MAX] ULP=0  (no differences found)")
    else:
        i = tracker.info
        print(
            f"[MAX] kind={i['kind']} chunk={i['chunk']} idx={i['idx']}  "
            f"u=0x{i['u']:08x} x={i['x']:.9g}  "
            f"A={i['A']!r} ({i['Abits']})  "
            f"B={i['B']!r} ({i['Bbits']})  "
            f"ULP={tracker.max_ulp}"
        )

    print(f"[done] mismatches printed: {total_printed}")

if __name__ == "__main__":
    main()
