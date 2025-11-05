#!/usr/bin/env python3
"""
调试测试 - 验证重力和物理
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src import PhysicsSimulator

def main():
    print("=" * 70)
    print("Physics Debug Test")
    print("=" * 70)
    
    # 简单配置 - 10个球
    NUM_OBJECTS = 10
    WORLD_BOUNDS = ((-10, 0, -10), (10, 20, 10))
    
    print(f"\nConfiguration:")
    print(f"  Objects: {NUM_OBJECTS}")
    print(f"  Gravity: (0, -9.8, 0)")
    
    # 初始化仿真器
    sim = PhysicsSimulator(
        num_objects=NUM_OBJECTS,
        world_bounds=WORLD_BOUNDS,
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0.0, -9.8, 0.0),
        damping=0.0
    )
    
    # 设置场景 - 简单的一列球
    positions = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
    for i in range(NUM_OBJECTS):
        positions[i] = [0, 15.0, 0]  # 所有球在同一位置开始
    
    sim.bodies.set_positions(positions)
    
    # 所有球相同参数
    radii = np.ones(NUM_OBJECTS, dtype=np.float32) * 0.3
    sim.bodies.set_radii(radii)
    
    masses = np.ones(NUM_OBJECTS, dtype=np.float32) * 1.0
    sim.bodies.set_masses(masses)
    
    restitutions = np.ones(NUM_OBJECTS, dtype=np.float32) * 0.8
    sim.bodies.set_restitutions(restitutions)
    
    velocities = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
    sim.bodies.set_velocities(velocities)
    
    print("\nInitial State:")
    data = sim.bodies.to_cpu()
    print(f"  Position[0]: {data['positions'][0]}")
    print(f"  Velocity[0]: {data['velocities'][0]}")
    print(f"  Mass[0]: {data['masses'][0]}")
    print(f"  dt: {sim.dt}")
    print(f"  gravity: {sim.gravity.get()}")
    
    # 运行10步
    print("\nRunning simulation for 10 steps...")
    for step in range(10):
        step_info = sim.step()
        
        if step % 3 == 0:
            data = sim.bodies.to_cpu()
            print(f"\nStep {step}:")
            print(f"  Position[0]: {data['positions'][0]}")
            print(f"  Velocity[0]: {data['velocities'][0]}")
            print(f"  Collisions: {step_info['num_collisions']}")
    
    print("\nFinal State:")
    data = sim.bodies.to_cpu()
    print(f"  Position[0]: {data['positions'][0]}")
    print(f"  Velocity[0]: {data['velocities'][0]}")
    
    # 检查是否下落
    final_y = data['positions'][0][1]
    initial_y = 15.0
    
    print(f"\nPhysics Check:")
    print(f"  Initial Y: {initial_y:.2f}")
    print(f"  Final Y: {final_y:.2f}")
    print(f"  Distance fallen: {initial_y - final_y:.2f}")
    
    if final_y < initial_y - 0.5:
        print("  ✓ Gravity is working!")
    else:
        print("  ✗ WARNING: Objects did not fall! Gravity may not be working.")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
