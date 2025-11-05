#!/usr/bin/env python3
"""
Direct kernel test - bypass all wrapper code
"""

import cupy as cp
import numpy as np

# Simplest possible gravity kernel
GRAVITY_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void apply_gravity(
    float* positions,
    float* velocities,
    int num_objects,
    float dt,
    float gravity_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // Update velocity
    float vy = velocities[idx * 3 + 1];
    vy += gravity_y * dt;
    velocities[idx * 3 + 1] = vy;
    
    // Update position
    float py = positions[idx * 3 + 1];
    py += vy * dt;
    positions[idx * 3 + 1] = py;
}
''', 'apply_gravity')

def main():
    print("Direct Kernel Test")
    print("=" * 50)
    
    # Create test data
    N = 1
    positions = cp.array([[0.0, 8.0, 0.0]], dtype=cp.float32)
    velocities = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    
    dt = np.float32(1.0 / 60.0)
    gravity_y = np.float32(-9.8)
    
    print(f"\nBefore:")
    print(f"  pos_y: {positions[0, 1].get()}")
    print(f"  vel_y: {velocities[0, 1].get()}")
    print(f"  dt: {dt}")
    print(f"  gravity_y: {gravity_y}")
    
    # Run 10 steps
    print(f"\nRunning 10 simulation steps...")
    for step in range(10):
        GRAVITY_KERNEL(
            (1,), (256,),
            (positions, velocities, N, dt, gravity_y)
        )
        cp.cuda.Stream.null.synchronize()
        
        if step % 3 == 0:
            print(f"  Step {step}: pos_y={positions[0, 1].get():.4f}, vel_y={velocities[0, 1].get():.4f}")
    
    print(f"\nAfter:")
    print(f"  pos_y: {positions[0, 1].get()}")
    print(f"  vel_y: {velocities[0, 1].get()}")
    
    # Check physics
    expected_vel = float(gravity_y * dt * 10)
    actual_vel = float(velocities[0, 1].get())
    
    print(f"\nExpected vel_y after 10 steps: {expected_vel:.4f}")
    print(f"Actual vel_y: {actual_vel:.4f}")
    print(f"Difference: {abs(expected_vel - actual_vel):.6f}")
    
    if abs(expected_vel - actual_vel) < 0.1:
        print("\n✓ Kernel works!")
    else:
        print("\n✗ Kernel FAILED!")

if __name__ == '__main__':
    main()
