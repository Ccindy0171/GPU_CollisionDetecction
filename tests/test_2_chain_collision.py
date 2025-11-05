#!/usr/bin/env python3
"""
Test 2: 3-ball chain collision
Ball 0 hits Ball 1 (stationary), Ball 1 should hit Ball 2 (stationary)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_chain_collision():
    """Test 3 balls in a chain collision"""
    print("=" * 70)
    print("TEST 2: CHAIN COLLISION (3 balls, no gravity)")
    print("=" * 70)
    
    sim = PhysicsSimulator(
        num_objects=3,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)
    )
    
    # Ball 0: moving right at 10 m/s
    # Ball 1: stationary, touching ball 2
    # Ball 2: stationary
    positions = np.array([
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.61, 0.0, 0.0],  # Just touching ball 1 (0.3 + 0.3 + 0.01)
    ], dtype=np.float32)
    
    velocities = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    restitutions = np.array([0.95, 0.95, 0.95], dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    print("\nINITIAL STATE:")
    print(f"  Ball 0: pos={positions[0]}, vel={velocities[0]} (MOVING)")
    print(f"  Ball 1: pos={positions[1]}, vel={velocities[1]} (STATIONARY)")
    print(f"  Ball 2: pos={positions[2]}, vel={velocities[2]} (STATIONARY)")
    print(f"  Distance 0-1: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"  Distance 1-2: {np.linalg.norm(positions[2] - positions[1]):.3f}m")
    
    collision_01 = False
    collision_12 = False
    ball2_moving = False
    
    print("\nSIMULATING...")
    for frame in range(100):
        vel_before = cp.asnumpy(sim.bodies.velocities)
        
        stats = sim.step()
        
        pos_after = cp.asnumpy(sim.bodies.positions)
        vel_after = cp.asnumpy(sim.bodies.velocities)
        
        # Check if ball 0 hit ball 1
        if not collision_01 and vel_after[1, 0] > 0.1:
            collision_01 = True
            print(f"\n  Frame {frame}: ✓ Ball 0 hit Ball 1")
            print(f"    Ball 0 vel: {vel_before[0]} -> {vel_after[0]}")
            print(f"    Ball 1 vel: {vel_before[1]} -> {vel_after[1]}")
            print(f"    Ball 2 vel: {vel_before[2]} -> {vel_after[2]}")
        
        # Check if ball 1 hit ball 2
        if collision_01 and not collision_12 and vel_after[2, 0] > 0.1:
            collision_12 = True
            print(f"\n  Frame {frame}: ✓ Ball 1 hit Ball 2")
            print(f"    Ball 0 vel: {vel_after[0]}")
            print(f"    Ball 1 vel: {vel_after[1]}")
            print(f"    Ball 2 vel: {vel_after[2]}")
        
        # Check if ball 2 is moving significantly
        if vel_after[2, 0] > 1.0:
            ball2_moving = True
        
        if collision_12 and ball2_moving:
            print(f"\n  Frame {frame}: ✓ Chain reaction complete")
            print(f"    Ball 2 is moving at {vel_after[2, 0]:.2f} m/s")
            break
        
        if frame % 20 == 0:
            print(f"  Frame {frame}: collisions={stats['num_collisions']}, "
                  f"vel=[{vel_after[0,0]:.2f}, {vel_after[1,0]:.2f}, {vel_after[2,0]:.2f}]")
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    if collision_01 and collision_12 and ball2_moving:
        print("✓✓✓ SUCCESS: Chain collision worked correctly!")
        print("    - Ball 0 hit Ball 1")
        print("    - Ball 1 hit Ball 2")
        print("    - Ball 2 moved away")
        return True
    else:
        print("✗✗✗ FAILURE:")
        if not collision_01:
            print("    - Ball 0 did NOT hit Ball 1")
        if not collision_12:
            print("    - Ball 1 did NOT hit Ball 2")
        if not ball2_moving:
            print("    - Ball 2 is NOT moving")
        return False

if __name__ == "__main__":
    success = test_chain_collision()
    sys.exit(0 if success else 1)
