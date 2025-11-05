#!/usr/bin/env python3
"""Test collision response with detailed position tracking"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_collision_response():
    """Test if collision response actually separates balls"""
    print("=" * 60)
    print("Collision Response Test")
    print("=" * 60)
    
    # Create 2 heavily overlapping balls
    num_balls = 2
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # No gravity to isolate collision effects
    )
    
    # Place balls very close together (heavily overlapping)
    positions = np.array([
        [0.0, 0.0, 0.0],   # Ball 1 at origin
        [0.3, 0.0, 0.0],   # Ball 2 only 0.3m away (overlapping by 0.3m!)
    ], dtype=np.float32)
    
    velocities = np.zeros((num_balls, 3), dtype=np.float32)
    radii = np.array([0.5, 0.5], dtype=np.float32)  # Each has radius 0.5
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitutions = np.array([0.8, 0.8], dtype=np.float32)
    
    # Initialize
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    print(f"\nInitial State:")
    print(f"  Ball 1: pos={positions[0]}, vel={velocities[0]}")
    print(f"  Ball 2: pos={positions[1]}, vel={velocities[1]}")
    print(f"  Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"  Sum of radii: {radii[0] + radii[1]:.3f}m")
    print(f"  Penetration: {(radii[0] + radii[1]) - np.linalg.norm(positions[1] - positions[0]):.3f}m")
    
    # Run several frames
    print(f"\nRunning 10 frames with collision detection/response:")
    for frame in range(10):
        # Build grid and detect collisions
        sim.build_grid()
        num_collisions = sim.detect_collisions()
        
        # Resolve collisions
        if num_collisions > 0:
            sim.resolve_collisions(num_collisions)
        
        # Get current state
        pos = cp.asnumpy(sim.bodies.positions)
        vel = cp.asnumpy(sim.bodies.velocities)
        dist = np.linalg.norm(pos[1] - pos[0])
        sum_r = radii[0] + radii[1]
        penetration = sum_r - dist
        
        print(f"\nFrame {frame}:")
        print(f"  Collisions detected: {num_collisions}")
        print(f"  Ball 1: pos=[{pos[0][0]:.4f}, {pos[0][1]:.4f}, {pos[0][2]:.4f}], "
              f"vel=[{vel[0][0]:.4f}, {vel[0][1]:.4f}, {vel[0][2]:.4f}]")
        print(f"  Ball 2: pos=[{pos[1][0]:.4f}, {pos[1][1]:.4f}, {pos[1][2]:.4f}], "
              f"vel=[{vel[1][0]:.4f}, {vel[1][1]:.4f}, {vel[1][2]:.4f}]")
        print(f"  Distance: {dist:.4f}m, Penetration: {penetration:.4f}m")
        
        # Integrate (even with no gravity, damping might affect velocities)
        sim.integrate()
    
    # Final check
    final_pos = cp.asnumpy(sim.bodies.positions)
    final_dist = np.linalg.norm(final_pos[1] - final_pos[0])
    final_penetration = sum_r - final_dist
    
    print(f"\n{'='*60}")
    print(f"Final Result:")
    print(f"  Final distance: {final_dist:.4f}m")
    print(f"  Required distance (sum of radii): {sum_r:.4f}m")
    print(f"  Final penetration: {final_penetration:.4f}m")
    
    if final_penetration <= 0.01:  # Allow 1cm tolerance
        print(f"\n✓ SUCCESS: Balls are properly separated!")
        return True
    else:
        print(f"\n✗ FAILURE: Balls are still penetrating by {final_penetration:.4f}m!")
        print(f"  This suggests collision response is too weak or not working.")
        return False

if __name__ == "__main__":
    success = test_collision_response()
    sys.exit(0 if success else 1)
