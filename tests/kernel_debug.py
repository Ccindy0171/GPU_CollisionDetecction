#!/usr/bin/env python3
"""
Kernel 调试测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.kernels import INTEGRATE_KERNEL

def main():
    print("=" * 70)
    print("Kernel Debug Test")
    print("=" * 70)
    
    # 创建简单测试数据
    N = 5
    
    positions = cp.array([
        [0, 10, 0],
        [1, 10, 1],
        [2, 10, 2],
        [3, 10, 3],
        [4, 10, 4]
    ], dtype=cp.float32)
    
    velocities = cp.zeros((N, 3), dtype=cp.float32)
    forces = cp.zeros((N, 3), dtype=cp.float32)
    masses = cp.ones(N, dtype=cp.float32)
    gravity = cp.array([0.0, -9.8, 0.0], dtype=cp.float32)
    radii = cp.ones(N, dtype=cp.float32) * 0.3
    restitutions = cp.ones(N, dtype=cp.float32) * 0.8
    bounds_min = cp.array([-10, 0, -10], dtype=cp.float32)
    bounds_max = cp.array([10, 20, 10], dtype=cp.float32)
    
    dt = 1.0 / 60.0
    damping = 0.01
    
    print("\n前:")
    print(f"  位置[0]: {positions[0].get()}")
    print(f"  速度[0]: {velocities[0].get()}")
    print(f"  重力: {gravity.get()}")
    print(f"  dt: {dt}")
    print(f"  质量[0]: {masses[0].get()}")
    
    # 调用kernel
    blocks = 1
    threads_per_block = 256
    
    INTEGRATE_KERNEL(
        (blocks,), (threads_per_block,),
        (
            positions,
            velocities,
            forces,
            masses,
            gravity,
            dt,
            damping,
            bounds_min,
            bounds_max,
            radii,
            restitutions,
            N
        )
    )
    
    cp.cuda.Stream.null.synchronize()
    
    print("\n后:")
    print(f"  位置[0]: {positions[0].get()}")
    print(f"  速度[0]: {velocities[0].get()}")
    
    # 计算期望值
    expected_vel_y = -9.8 * dt
    expected_pos_y = 10 + expected_vel_y * dt
    
    print(f"\n期望:")
    print(f"  速度 Y: {expected_vel_y:.6f}")
    print(f"  位置 Y: {expected_pos_y:.6f}")
    
    actual_vel_y = velocities[0, 1].get()
    actual_pos_y = positions[0, 1].get()
    
    print(f"\n实际:")
    print(f"  速度 Y: {actual_vel_y:.6f}")
    print(f"  位置 Y: {actual_pos_y:.6f}")
    
    vel_diff = abs(actual_vel_y - expected_vel_y)
    pos_diff = abs(actual_pos_y - expected_pos_y)
    
    print(f"\n差异:")
    print(f"  速度差异: {vel_diff:.8f}")
    print(f"  位置差异: {pos_diff:.8f}")
    
    if vel_diff < 0.01 and pos_diff < 0.01:
        print("\n✓ Kernel 工作正常！")
    else:
        print("\n✗ Kernel 有问题！")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
