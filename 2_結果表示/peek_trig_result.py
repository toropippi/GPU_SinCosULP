#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCLで生成した 2^24 要素/チャンクの出力を、入力x（float or 0x...）から一点参照するツール。
ファイル命名: "<chunk>_sin", "<chunk>_cos", "<chunk>_native_sin", "<chunk>_native_cos"
各ファイルは float32 リトルエンディアンの生配列（長さ 2^24 = 16,777,216）。

使い方:
  python peek_trig_result.py --dir <出力フォルダ>
"""

import argparse
import os
import numpy as np

CHUNK_LOG2 = 24
CHUNK_SIZE = 1 << CHUNK_LOG2

KINDS = ("sin", "cos", "native_sin", "native_cos")  # 読み出す4種類
DTYPE = "<f4"  # little-endian float32

# 縮約用定数（float32 で扱う）
INV_TAU_F32 = np.float32(0.15915493667125701904)  # 1/(2π)
TAU_F32     = np.float32(1.0) / INV_TAU_F32       # 2π（掛け算で使用）

REDUCE_KIND = "native_sin"  # 縮約後の入力を与える先

# ---------------------------
# 低レベル/ユーティリティ
# ---------------------------

def parse_hex_u32(token: str) -> int:
    s = token.strip().lower().replace("_", "")
    if not s.startswith("0x"):
        raise ValueError("hexは 0x で始めてください")
    u = int(s, 16)
    if not (0 <= u <= 0xFFFFFFFF):
        raise ValueError("u32範囲外")
    return u

def float_to_u32_and_f32(val: float) -> tuple[int, np.float32]:
    x32 = np.float32(val)
    u32 = x32.view(np.uint32).item()
    return u32, x32

def u32_to_f32(u: int) -> np.float32:
    return np.array([u], dtype=np.uint32).view(np.float32)[0]

def bits_to_hex(u: int) -> str:
    return f"0x{u:08x}"

def f32_to_bits_hex(x: np.float32) -> str:
    return bits_to_hex(np.array([x], dtype=np.float32).view(np.uint32)[0])

def path_for_chunk_kind(root: str, chunk_hi: int, kind: str) -> str:
    return os.path.join(root, f"{chunk_hi}_{kind}")

def read_one_value(path: str, idx: int) -> tuple[float, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mm = np.memmap(path, dtype=DTYPE, mode="r", shape=(CHUNK_SIZE,))
    if not (0 <= idx < CHUNK_SIZE):
        del mm
        raise IndexError("idx out of range")
    v32 = np.float32(mm[idx])
    bits_hex = f32_to_bits_hex(v32)
    val = float(v32)
    del mm
    return val, bits_hex

def resolve_u_from_input(token: str) -> tuple[int, np.float32]:
    t = token.strip()
    if t.lower().startswith("0x"):
        u = parse_hex_u32(t)
        x32 = u32_to_f32(u)
        return u, x32
    try:
        f = float(t)
    except ValueError as e:
        raise ValueError("数値（例: 1.0）または 0xXXXXXXXX を入力してください") from e
    u, x32 = float_to_u32_and_f32(f)
    return u, x32

def u32_to_chunk_idx(u: int) -> tuple[int, int]:
    return (u >> CHUNK_LOG2) & 0xFF, (u & (CHUNK_SIZE - 1))

# ---------------------------
# RZ 乗算（FMUL.RZ 相当のエミュ）
# ---------------------------

def f32_mul_rz(a32: np.float32, b32: np.float32) -> np.float32:
    """
    a*b を float64 で計算 → float32 へ RN 変換 → もし RN が 0 から遠ざかっていたら
    1ULP だけ 0 方向に nextafter して RZ を再現。
    """
    if not (np.isfinite(a32) and np.isfinite(b32)):
        return np.float32(np.float64(a32) * np.float64(b32))
    y64 = np.float64(a32) * np.float64(b32)           # 実数積の高精度近似
    f = np.float32(y64)                                # RN
    # RN が 0 から遠ざかったケースのみ 1ULP だけ戻す
    if y64 > 0.0 and f > y64:
        f = np.nextafter(f, np.float32(0.0))
    elif y64 < 0.0 and f < y64:
        f = np.nextafter(f, np.float32(0.0))
    # y64 == 0.0 or NaN の場合はそのまま
    return np.float32(f)

# ---------------------------
# ±2π 縮約（RN/RZ 切替）
# ---------------------------

def reduce_pm2pi(x32: np.float32, mode: str = "rn") -> np.float32:
    """
    |x| > 2π のとき、float32 精度で ±2π 未満へ縮約。
      mode='rn' : 既存（x*INV_TAU を RN）
      mode='rz' : 乗算のみ RZ（FMUL.RZ 相当）で実施
    """
    if not np.isfinite(x32):
        return x32
    if np.abs(x32) <= TAU_F32:
        return x32

    if mode == "rz":
        t = f32_mul_rz(x32, INV_TAU_F32)
    else:
        t = np.float32(x32 * INV_TAU_F32)  # RN

    t_trunc = np.float32(np.trunc(t))      # 0 方向で整数部を落とす
    fract   = np.float32(t - t_trunc)
    reduced = np.float32(fract * TAU_F32)  # 2π を掛けて元スケールへ
    return reduced

# ---------------------------
# メイン
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenCL trig sweep viewer")
    ap.add_argument("--dir", default=".", help="出力ファイルのディレクトリ（デフォルト: カレント）")
    args = ap.parse_args()
    root = os.path.abspath(args.dir)

    print(f"[info] dir = {root}")
    print("[info] ファイル命名例: <chunk>_sin / <chunk>_cos / <chunk>_native_sin / <chunk>_native_cos")
    print("[info] 1ディスパッチ=2^24 要素、chunk_hi=0..255, idx=0..(2^24-1)")
    print("  入力例: 0x3f800000 / 1.0 / -0.0 / nan / inf / -inf")
    print("  終了: q または quit")

    while True:
        try:
            token = input("\nx> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if token.lower() in ("q", "quit", "exit"):
            break
        if token == "":
            continue

        try:
            u, x32 = resolve_u_from_input(token)
        except Exception as e:
            print(f"[error] {e}")
            continue

        chunk_hi, idx = u32_to_chunk_idx(u)

        print(f"  input  : token='{token}'")
        print(f"  float32: {x32}  (bits={f32_to_bits_hex(x32)})")
        print(f"  u32    : {bits_to_hex(u)}")
        print(f"  chunk  : {chunk_hi} (0x{chunk_hi:02x}), idx={idx} (0x{idx:06x})")

        # 各ファイルから一点参照
        for kind in KINDS:
            path = path_for_chunk_kind(root, chunk_hi, kind)
            try:
                val, bits_hex = read_one_value(path, idx)
                print(f"  {kind:<11}: {val!r:<16}  bits={bits_hex}  ({os.path.basename(path)})")
            except FileNotFoundError:
                print(f"  {kind:<11}: [missing file] {os.path.basename(path)}")
            except Exception as e:
                print(f"  {kind:<11}: [error] {e}")

        # 追加: ±2π縮約（RN / RZ の比較）
        if np.isfinite(x32):
            # RN
            x_rn = reduce_pm2pi(x32, mode="rn")
            u_rn = np.array([x_rn], dtype=np.float32).view(np.uint32)[0].item()
            chi_rn, idx_rn = u32_to_chunk_idx(u_rn)
            path_rn = path_for_chunk_kind(root, chi_rn, REDUCE_KIND)

            # RZ
            x_rz = reduce_pm2pi(x32, mode="rz")
            u_rz = np.array([x_rz], dtype=np.float32).view(np.uint32)[0].item()
            chi_rz, idx_rz = u32_to_chunk_idx(u_rz)
            path_rz = path_for_chunk_kind(root, chi_rz, REDUCE_KIND)

            # RN 行
            try:
                val_rn, bits_rn_val = read_one_value(path_rn, idx_rn)
                print(f"  reduce±2pi[RN]→{REDUCE_KIND}: {val_rn!r:<16}  bits={bits_rn_val}  ({os.path.basename(path_rn)}; x'={x_rn} bits={f32_to_bits_hex(x_rn)})")
            except FileNotFoundError:
                print(f"  reduce±2pi[RN]→{REDUCE_KIND}: [missing file] {os.path.basename(path_rn)}  (x'={x_rn} bits={f32_to_bits_hex(x_rn)})")
                val_rn = None
            except Exception as e:
                print(f"  reduce±2pi[RN]→{REDUCE_KIND}: [error] {e}  (x'={x_rn} bits={f32_to_bits_hex(x_rn)})")
                val_rn = None

            # RZ 行
            try:
                val_rz, bits_rz_val = read_one_value(path_rz, idx_rz)
                print(f"  reduce±2pi[RZ]→{REDUCE_KIND}: {val_rz!r:<16}  bits={bits_rz_val}  ({os.path.basename(path_rz)}; x'={x_rz} bits={f32_to_bits_hex(x_rz)})")
            except FileNotFoundError:
                print(f"  reduce±2pi[RZ]→{REDUCE_KIND}: [missing file] {os.path.basename(path_rz)}  (x'={x_rz} bits={f32_to_bits_hex(x_rz)})")
                val_rz = None
            except Exception as e:
                print(f"  reduce±2pi[RZ]→{REDUCE_KIND}: [error] {e}  (x'={x_rz} bits={f32_to_bits_hex(x_rz)})")
                val_rz = None

            # 差分
            if (val_rn is not None) and (val_rz is not None):
                diff = np.float64(val_rz) - np.float64(val_rn)
                print(f"  diff(RZ−RN): {diff:+.9e}")
        else:
            print(f"  reduce±2pi→{REDUCE_KIND}: [skipped: non-finite input]")

if __name__ == "__main__":
    main()
