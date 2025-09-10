#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peek+compare (ordered print for human reading):
  入力した x（float or 0x...）に対応する 1 点を、
  TEST(--dir) の {sin,cos,native_sin,native_cos} と
  REF(--ref; 既定 ./doubleSample) の {sin,cos} から読み出し、
  sin / cos を分けて各3行（REF→TEST→TEST.native_*）で表示する。

ファイル仕様:
  <chunk>_{sin|cos|native_sin|native_cos} / <chunk>_{sin|cos} （各 2^24 要素, float32 LE）

使い方:
  python peek_compare_to_ref_pretty.py --dir <test_dir> [--ref doubleSample]
"""

import argparse
import os
import numpy as np

CHUNK_LOG2 = 24
CHUNK_SIZE = 1 << CHUNK_LOG2
DTYPE = "<f4"

TEST_KINDS = ("sin", "cos", "native_sin", "native_cos")
REF_KIND_OF = {"sin":"sin", "native_sin":"sin", "cos":"cos", "native_cos":"cos"}

# ---------- 基本ユーティリティ ----------
def parse_hex_u32(token: str) -> int:
    s = token.strip().lower().replace("_", "")
    if not s.startswith("0x"):
        raise ValueError("hexは 0x で始めてください")
    u = int(s, 16)
    if not (0 <= u <= 0xffffffff):
        raise ValueError("u32範囲外")
    return u

def float_to_u32_and_f32(val: float) -> tuple[int, np.float32]:
    x32 = np.float32(val)
    u32 = x32.view(np.uint32).item()
    return u32, x32

def u32_to_f32(u: int) -> np.float32:
    return np.array([u], dtype=np.uint32).view(np.float32)[0]

def f32_bits_hex(x: np.float32) -> str:
    return f"0x{np.array([x], dtype=np.float32).view(np.uint32)[0]:08x}"

def u_to_chunk_idx(u: int) -> tuple[int, int]:
    return (u >> CHUNK_LOG2) & 0xFF, (u & (CHUNK_SIZE - 1))

def open_chunk(root: str, chunk: int, kind: str) -> np.memmap:
    path = os.path.join(root, f"{chunk}_{kind}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # サイズチェック（2^24 * 4 bytes）
    expected = CHUNK_SIZE * 4
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(f"size mismatch: {path} ({actual} != {expected})")
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))

def read_one(root: str, chunk: int, kind: str, idx: int):
    """1点だけ読み出し。成功なら (value(np.float32), bits_hex) を返す。"""
    mm = open_chunk(root, chunk, kind)
    v32 = np.float32(mm[idx])
    del mm
    return v32, f32_bits_hex(v32)

def resolve_input(token: str) -> tuple[int, np.float32]:
    t = token.strip()
    if t.lower().startswith("0x"):
        u = parse_hex_u32(t)
        return u, u32_to_f32(u)
    try:
        f = float(t)
    except ValueError as e:
        raise ValueError("数値（例: 1.0）または 0xXXXXXXXX を入力してください") from e
    return float_to_u32_and_f32(f)
# ---------- ULP（スカラー用；nativeだけに表示） ----------
def _to_ordered_u32_scalar(u: np.uint32) -> np.uint32:
    # IEEE754 float32 を単調な整数順序に写像（負側の順序反転を補正）
    sign = (u >> np.uint32(31))
    return (u | np.uint32(0x80000000)) if sign == 0 else np.uint32(~u)

def ulp_distance_scalar(a: np.float32, b: np.float32,
                        treat_zero_equal: bool = True,
                        treat_nan_equal: bool = True) -> int:
    """a と b の ULP距離を返す（+0/-0 と NaN同士は一致扱いが既定）"""
    if treat_nan_equal and np.isnan(a) and np.isnan(b):
        return 0
    if treat_zero_equal and (a == 0.0) and (b == 0.0):
        return 0
    ua = np.array([a], dtype=np.float32).view(np.uint32)[0]
    ub = np.array([b], dtype=np.float32).view(np.uint32)[0]
    oa = np.int64(_to_ordered_u32_scalar(ua))
    ob = np.int64(_to_ordered_u32_scalar(ub))
    return int(abs(oa - ob))
# ---------- 表示ヘルパ ----------
def print_line(label: str, value_tuple, path: str):
    if isinstance(value_tuple, tuple):
        val, bits = value_tuple
        print(f"  {label:<12} {float(val)!r:<20} bits={bits}  ({os.path.basename(path)})")
    else:
        # value_tuple はエラーメッセージ文字列
        print(f"  {label:<12} {value_tuple}")

def fetch_or_msg(root: str, chunk: int, kind: str, idx: int):
    path = os.path.join(root, f"{chunk}_{kind}")
    try:
        v = read_one(root, chunk, kind, idx)
        return v, path
    except FileNotFoundError:
        return f"[missing file] {os.path.basename(path)}", path
    except Exception as e:
        return f"[error] {e}", path

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser(description="Peek one value and print REF/TEST/native in ordered layout (sin & cos).")
    ap.add_argument("--dir", required=True, help="TEST ディレクトリ（0..255_{sin,cos,native_sin,native_cos}）")
    ap.add_argument("--ref", default="doubleSample", help="REF ディレクトリ（0..255_{sin,cos}）")
    args = ap.parse_args()

    test_root = os.path.abspath(args.dir)
    ref_root  = os.path.abspath(args.ref)

    print(f"[info] TEST={test_root}")
    print(f"[info] REF ={ref_root}")
    print("[info] ファイル命名: <chunk>_sin / _cos / _native_sin / _native_cos（TEST側）、<chunk>_sin / _cos（REF側）")
    print("[info] 1ディスパッチ=2^24 要素、chunk=0..255, idx=0..(2^24-1)")
    print("  入力例: 0x3f800000 / 1.0 / -0.0 / nan / inf / -inf")
    print("  終了: q / quit / exit")

    while True:
        try:
            token = input("\nx> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if token.lower() in ("q","quit","exit"):
            break
        if token == "":
            continue

        # 入力解決
        try:
            u, x32 = resolve_input(token)
        except Exception as e:
            print(f"[error] {e}")
            continue

        chunk, idx = u_to_chunk_idx(u)
        print(f"  input  : token='{token}'")
        print(f"  float32: {x32}  (bits={f32_bits_hex(x32)})")
        print(f"  u32    : 0x{u:08x}")

        # ---- sin セクション ----
        print("\n[sin]")
        ref_val, ref_path = fetch_or_msg(ref_root,  chunk, "sin", idx)
        test_val, test_path = fetch_or_msg(test_root, chunk, "sin", idx)
        natv_val, natv_path = fetch_or_msg(test_root, chunk, "native_sin", idx)

        print_line("REF.sin:", ref_val, ref_path)
        print_line("TEST.sin:", test_val, test_path)
        if isinstance(natv_val, tuple) and isinstance(ref_val, tuple):
            natv_v, natv_bits = natv_val
            ref_v, _ = ref_val
            ulp = ulp_distance_scalar(natv_v, ref_v, treat_zero_equal=True, treat_nan_equal=True)
            print(f"  {'TEST.native:':<12} {float(natv_v)!r:<20} bits={natv_bits}  (ULP={ulp})  ({os.path.basename(natv_path)})")
        else:
            print_line("TEST.native:", natv_val, natv_path)

        # ---- cos セクション ----
        print("\n[cos]")
        refc_val, refc_path = fetch_or_msg(ref_root,  chunk, "cos", idx)
        testc_val, testc_path = fetch_or_msg(test_root, chunk, "cos", idx)
        natvc_val, natvc_path = fetch_or_msg(test_root, chunk, "native_cos", idx)

        print_line("REF.cos:", refc_val, refc_path)
        print_line("TEST.cos:", testc_val, testc_path)
        if isinstance(natvc_val, tuple) and isinstance(refc_val, tuple):
            natc_v, natc_bits = natvc_val
            refc_v, _ = refc_val
            ulp = ulp_distance_scalar(natc_v, refc_v, treat_zero_equal=True, treat_nan_equal=True)
            print(f"  {'TEST.native:':<12} {float(natc_v)!r:<20} bits={natc_bits}  (ULP={ulp})  ({os.path.basename(natvc_path)})")
        else:
            print_line("TEST.native:", natvc_val, natvc_path)

if __name__ == "__main__":
    main()
