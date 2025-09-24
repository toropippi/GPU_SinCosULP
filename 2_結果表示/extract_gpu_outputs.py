#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
args.txt の各引数について、3 GPU ディレクトリから native_sin の値を取り出し、
TSV に出力する。cos は扱わない。各値は数値と u32 16進表記も出力する。

出力列:
  arg_str, arg_f32, arg_hex,
  sin_gfx1036_win, sin_hex_gfx1036_win,
  sin_intelUHD770_win, sin_hex_intelUHD770_win,
  sin_RTX5090_win, sin_hex_RTX5090_win
"""
from __future__ import annotations
import argparse, os, sys, re
from typing import Dict, Tuple, List
import numpy as np

U24_MASK = (1 << 24) - 1
F32_TINY = np.finfo(np.float32).tiny  # ~1.1754944e-38

DEFAULT_GPUS = ["gfx1036_win", "intelUHD770_win", "RTX5090_win"]
# cos を削除し、デフォルトは native_sin のみ
DEFAULT_OPS  = ["native_sin"]

HEX_RE = re.compile(r"^\s*0x([0-9a-fA-F]{8})\s*$")

def f32_to_u32(x: np.float32) -> int:
    return int(np.frombuffer(np.float32(x).tobytes(), dtype="<u4")[0])

def u32_to_f32(u: int) -> np.float32:
    return np.frombuffer(np.uint32(u).tobytes(), dtype="<f4")[0]

def parse_arg_line(line: str):
    s = line.strip()
    if not s:
        raise ValueError("empty line")
    m = HEX_RE.match(s)
    if m:
        u = int(m.group(1), 16)
        x = u32_to_f32(u)
        return s, np.float32(x), u
    try:
        x = np.float32(float(s))
    except Exception as e:
        raise ValueError(f"invalid number: {s}") from e
    u = f32_to_u32(x)
    return s, x, u

def is_subnormal_f32(x: np.float32) -> bool:
    if x == np.float32(0.0):
        return False
    ax = abs(float(x))
    return (ax < F32_TINY) and np.isfinite(x)

class ChunkCache:
    def __init__(self, root: str, dtype="<f4"):
        self.root = root
        self.dtype = dtype
        self._cache: Dict[Tuple[str, str, int], np.memmap] = {}

    def get(self, gpu: str, op: str, chunk: int) -> np.memmap:
        key = (gpu, op, chunk)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.root, gpu, f"{chunk}_{op}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing file: {path}")
        mm = np.memmap(path, dtype=self.dtype, mode="r", shape=(1<<24,))
        self._cache[key] = mm
        return mm

def format_f32_hex(x: np.float32) -> str:
    return f"0x{f32_to_u32(x):08X}"

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--args-file", required=True, help="引数リスト（1行1引数: 10進 or 0x────────）")
    ap.add_argument("--root", default=".", help="GPU ディレクトリを含むルートパス")
    ap.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS, help="GPU ディレクトリ名のリスト")
    # cos は使わないが、将来拡張用に残す（既定は native_sin のみ）
    ap.add_argument("--ops",  nargs="+", default=DEFAULT_OPS, help="参照する関数名（既定: native_sin）")
    ap.add_argument("--output", default="outputs.tsv", help="集計出力（TSV）パス")
    ap.add_argument("--per-arg-files", action="store_true", help="各引数ごとの個別テキストも出力")
    ap.add_argument("--allow-subnormals", action="store_true", default=True, help="非正規化数も許容（±0は常に許容）")
    ap.add_argument("--include-naninf", action="store_true", help="NaN/Inf も許容（既定は除外）")
    ap.add_argument("--skip-missing", action="store_true", help="欠損時に NaN で継続（既定は例外）")
    args = ap.parse_args()

    # 明示的に native_sin のみに制限（--ops が与えられても native_sin 以外は無視）
    ops = [op for op in args.ops if op == "native_sin"]
    if not ops:
        ops = ["native_sin"]

    for gpu in args.gpus:
        d = os.path.join(args.root, gpu)
        if not os.path.isdir(d):
            print(f"[WARN] GPU dir not found: {d}", file=sys.stderr)

    cache = ChunkCache(args.root)

    # 出力ヘッダ（chunk/index は出力しない）
    # 各 GPU につき: sin_<gpu>, sin_hex_<gpu>
    header_cols: List[str] = ["arg_str", "arg_f32", "arg_hex"]
    for gpu in args.gpus:
        header_cols.append(f"sin_{gpu}")
        header_cols.append(f"sin_hex_{gpu}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "w", encoding="utf-8")
    out_f.write("\t".join(header_cols) + "\n")

    if args.per_arg_files:
        os.makedirs("per_arg", exist_ok=True)

    n_total = n_used = n_skipped = 0

    with open(args.args_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n_total += 1
            try:
                arg_str, x_f32, u32 = parse_arg_line(s)
            except Exception as e:
                print(f"[WARN] line {line_no}: {e}", file=sys.stderr)
                n_skipped += 1
                continue

            finite = np.isfinite(x_f32)
            subn   = is_subnormal_f32(x_f32)
            if not finite and not args.include_naninf:
                print(f"[SKIP] line {line_no}: non-finite (arg={arg_str})", file=sys.stderr)
                n_skipped += 1
                continue
            if subn and not args.allow_subnormals:
                print(f"[SKIP] line {line_no}: subnormal (arg={arg_str})", file=sys.stderr)
                n_skipped += 1
                continue

            # u32 ビット→ chunk / index
            chunk = (u32 >> 24) & 0xFF
            index = (u32 & U24_MASK)

            # native_sin を各 GPU から取得
            values_by_gpu: Dict[str, np.float32] = {}
            for gpu in args.gpus:
                try:
                    arr = cache.get(gpu, "native_sin", chunk)
                    values_by_gpu[gpu] = np.float32(arr[index])
                except Exception as e:
                    msg = f"[{'SKIP' if args.skip_missing else 'ERROR'}] {gpu}/{chunk}_native_sin: {e}"
                    print(msg, file=sys.stderr)
                    if args.skip_missing:
                        values_by_gpu[gpu] = np.float32(np.nan)
                    else:
                        raise

            # 行の生成（arg 情報＋各 GPU の数値/hex）
            row: List[str] = [
                arg_str,
                repr(float(x_f32)),
                f"0x{u32:08X}",
            ]
            for gpu in args.gpus:
                v = values_by_gpu.get(gpu, np.float32(np.nan))
                row.append(repr(float(v)))
                row.append(f"{format_f32_hex(np.float32(v)) if np.isfinite(v) else 'NaN'}")

            out_f.write("\t".join(row) + "\n")

            # per-arg ファイル（必要時）
            if args.per_arg_files:
                path = os.path.join("per_arg", f"arg_{u32:08X}.txt")
                with open(path, "w", encoding="utf-8") as pf:
                    pf.write(f"arg_str : {arg_str}\n")
                    pf.write(f"arg_f32 : {float(x_f32)}\n")
                    pf.write(f"arg_hex : 0x{u32:08X}\n")
                    pf.write("\n[native_sin]\n")
                    for gpu in args.gpus:
                        v = values_by_gpu.get(gpu, np.float32(np.nan))
                        if np.isfinite(v):
                            pf.write(f"  {gpu:16s}: {float(v):.9g}  ({format_f32_hex(np.float32(v))})\n")
                        else:
                            pf.write(f"  {gpu:16s}: NaN\n")

            n_used += 1

    out_f.close()
    print(f"[DONE] total lines: {n_total}, used: {n_used}, skipped: {n_skipped}")
    print(f"[OUT ] {os.path.abspath(args.output)}")
    if args.per_arg_files:
        print(f"[OUT ] per-arg files under ./per_arg/")

if __name__ == "__main__":
    main()
