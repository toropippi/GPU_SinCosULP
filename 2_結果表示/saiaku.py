#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

INV_TAU_F32 = np.float32(0.15915493667125701904)  # 1/(2π) (あなたが使っている定数)
TAU_F32     = np.float32(1.0) / INV_TAU_F32       # 2π（同じ定数から再構成）

def f32(x): return np.float32(x)
def bits(x32): return np.array([x32], dtype=np.float32).view(np.uint32)[0]
def hexbits(x32): return f"0x{bits(x32):08x}"

def f32_nextup(x):   return np.nextafter(f32(x),  f32(np.inf))
def f32_nextdown(x): return np.nextafter(f32(x),  f32(-np.inf))

def f32_mul_rz(a32, b32):
    """ a*b を f64 で計算→f32(RN)→0方向に 1ULP だけ寄せて RZ を再現 """
    y64 = np.float64(a32) * np.float64(b32)
    f   = f32(y64)  # RN
    if np.isfinite(y64) and y64 > 0.0 and f > y64:
        f = np.nextafter(f, f32(0.0))  # 1ULP だけ 0 方向へ
    elif np.isfinite(y64) and y64 < 0.0 and f < y64:
        f = np.nextafter(f, f32(0.0))
    return f32(f)

def k_from_x(x32, mode="rn"):
    """ k = trunc(x*(1/2π)) を返す（mode: rn/rz） """
    if mode == "rz":
        t = f32_mul_rz(x32, INV_TAU_F32)
    else:
        t = f32(x32 * INV_TAU_F32)  # RN
    return int(np.trunc(t)), t

def find_worst(sign=+1, N_max=1_000_000, limit=20):
    """
    sign=+1: +N·2π 境界の直前/直後を走査して RN 跨ぎを探す
    sign=-1: -N·2π について同様
    """
    found = 0
    sgn = 1 if sign >= 0 else -1
    for N in range(1, N_max+1):
        boundary = f32(sgn * N) * TAU_F32  # N*2π (float32)
        # 境界の前後を候補にする（両側を見る）
        for toward in (-np.inf, np.inf, 0.0):
            x = np.nextafter(boundary, f32(toward))
            k_rn, t_rn = k_from_x(x, "rn")
            k_rz, t_rz = k_from_x(x, "rz")

            # 期待する“最悪ケース”：RN が跨ぎ、RZ は跨がない
            if sgn > 0:
                ok = (k_rz == (N-1)) and (k_rn == N)
            else:
                ok = (k_rz == -(N-1)) and (k_rn == -N)

            if ok:
                # t の真値に近い f64 積も参考出力（丸め前）
                t64 = np.float64(x) * np.float64(INV_TAU_F32)
                # 情報表示
                print(f"N={N:6d} sign={sgn:+d}  x={x:.9f}  bits={hexbits(x)}")
                print(f"    t_RN={t_rn:.9f} -> k_RN={k_rn:>+4d}   "
                      f"t_RZ={t_rz:.9f} -> k_RZ={k_rz:>+4d}   "
                      f"(t64={t64:.12f})")
                print()
                found += 1
                break  # この N はもうヒットしたので次の N へ
        if found >= limit:
            break

if __name__ == "__main__":
    # 例：正の側で 10 件、負の側で 10 件
    print("=== Positive side (sign=+1) ===")
    find_worst(sign=+1, N_max=500000, limit=10)
    print("=== Negative side (sign=-1) ===")
    find_worst(sign=-1, N_max=500000, limit=10)
