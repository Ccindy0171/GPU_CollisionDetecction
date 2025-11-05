#!/usr/bin/env python3
"""
Test our actual INTEGRATE_KERNEL
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.kernels import INTEGRATE_KERNEL

def main():
    print("INTEGRATE_KERNEL Direct Test")
    print("=" * 60)
    
    # Create test data exactly like simulator does
    N = 1
    positions = cp.array([[0.0, 8.0, 0.0]], dtype=cp.float32)
    velocities = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    forces = cp.zeros((N, 3), dtype=cp.float32)
    masses = cp.ones(N, dtype=cp.float32)
    radii = cp.ones(N, dtype=cp.float32) * 0.3
    restitutions = cp.ones(N, dtype=cp.float32) * 0.8
    
    # Use CuPy array for gravity like simulator does
    gravity = cp.array([0.0, -9.8, 0.0], dtype=cp.float32)
    
    bounds_min = cp.array([-10.0, 0.0, -10.0], dtype=cp.float32)
    bounds_max = cp.array([10.0, 20.0, 10.0], dtype=cp.float32)
    
    dt = float(1.0 / 60.0)  # Python float
    damping = float(0.0)     # Python float
    
    print(f"\nInitial state:")
    print(f"  pos_y: {positions[0, 1].get()}")
    print(f"  vel_y: {velocities[0, 1].get()}")
    print(f"  gravity: {gravity.get()}")
    print(f"  dt: {dt} (type: {type(dt).__name__})")
    print(f"  damping: {damping} (type: {type(damping).__name__})")
    print(f"  forces: {forces[0].get()}")
    print(f"  mass: {masses[0].get()}")
    
    # Run 10 steps
    print(f"\nRunning 10 simulation steps...")
    for step in range(10):
        # Call kernel exactly like simulator does
        blocks = (N + 255) // 256
        threads = 256
        
        INTEGRATE_KERNEL(
            (blocks,), (threads,),
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
        
        if step % 3 == 0:
            pos_y = positions[0, 1].get()
            vel_y = velocities[0, 1].get()
            print(f"  Step {step:2d}: pos_y={pos_y:7.4f}, vel_y={vel_y:7.4f}")
    
    print(f"\nFinal state:")
    print(f"  pos_y: {positions[0, 1].get()}")
    print(f"  vel_y: {velocities[0, 1].get()}")
    
    # Physics check
    final_vel_y = velocities[0, 1].get()
    expected_vel_y = -9.8 * dt * 10
    
    print(f"\nPhysics verification:")
    print(f"  Expected vel_y: {expected_vel_y:.4f}")
    print(f"  Actual vel_y: {final_vel_y:.4f}")
    print(f"  Difference: {abs(final_vel_y - expected_vel_y):.6f}")
    
    if abs(final_vel_y - expected_vel_y) < 0.1:
        print("\n✓ INTEGRATE_KERNEL works!")
    else:
        print("\n✗ INTEGRATE_KERNEL FAILED!")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
