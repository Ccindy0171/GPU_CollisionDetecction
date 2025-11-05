#!/usr/bin/env python3
"""
Test 1: Head-on collision of 2 balls
Most basic test - no gravity, balls moving directly towards each other
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_head_on_collision():
    """Test 2 balls colliding head-on"""
    print("=" * 70)
    print("TEST 1: HEAD-ON COLLISION (2 balls, no gravity)")
    print("=" * 70)
    
    # Create simulator with NO gravity
    sim = PhysicsSimulator(
        num_objects=2,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # NO GRAVITY
    )
    
    # Ball 1 at x=-2, moving RIGHT (+x) at 5 m/s
    # Ball 2 at x=+2, moving LEFT (-x) at 5 m/s
    # They should collide near x=0
    positions = np.array([
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    velocities = np.array([
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitutions = np.array([1.0, 1.0], dtype=np.float32)  # Perfect elastic
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    print("\nINITIAL STATE:")
    print(f"  Ball 0: pos={positions[0]}, vel={velocities[0]}")
    print(f"  Ball 1: pos={positions[1]}, vel={velocities[1]}")
    print(f"  Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"  Min distance (sum of radii): {radii[0] + radii[1]:.3f}m")
    
    collision_detected = False
    velocities_reversed = False
    
    print("\nSIMULATING...")
    for frame in range(100):
        pos_before = cp.asnumpy(sim.bodies.positions)
        vel_before = cp.asnumpy(sim.bodies.velocities)
        
        stats = sim.step()
        
        pos_after = cp.asnumpy(sim.bodies.positions)
        vel_after = cp.asnumpy(sim.bodies.velocities)
        
        dist = np.linalg.norm(pos_after[1] - pos_after[0])
        
        if stats['num_collisions'] > 0 and not collision_detected:
            collision_detected = True
            print(f"\n  Frame {frame}: ✓ COLLISION DETECTED")
            print(f"    Position 0: {pos_after[0]}")
            print(f"    Position 1: {pos_after[1]}")
            print(f"    Distance: {dist:.4f}m")
            print(f"    Velocity 0 BEFORE: {vel_before[0]}")
            print(f"    Velocity 0 AFTER:  {vel_after[0]}")
            print(f"    Velocity 1 BEFORE: {vel_before[1]}")
            print(f"    Velocity 1 AFTER:  {vel_after[1]}")
            
            # Check if velocities reversed
            if vel_after[0, 0] < 0 and vel_after[1, 0] > 0:
                velocities_reversed = True
                print(f"    ✓ Velocities reversed correctly")
            else:
                print(f"    ✗ Velocities NOT reversed!")
        
        # Check if balls passed through each other
        if pos_before[0, 0] < pos_before[1, 0]:  # Ball 0 on left initially
            if pos_after[0, 0] > pos_after[1, 0] + 0.1:  # Ball 0 now on right
                if not collision_detected:
                    print(f"\n  Frame {frame}: ✗ TUNNELING - balls passed through!")
                    print(f"    Ball 0: {pos_before[0, 0]:.3f} -> {pos_after[0, 0]:.3f}")
                    print(f"    Ball 1: {pos_before[1, 0]:.3f} -> {pos_after[1, 0]:.3f}")
                    break
        
        # If collision happened and balls separated, test complete
        if collision_detected and dist > (radii[0] + radii[1]) * 1.5:
            print(f"\n  Frame {frame}: ✓ Balls separated")
            print(f"    Final distance: {dist:.3f}m")
            print(f"    Final velocity 0: {vel_after[0]}")
            print(f"    Final velocity 1: {vel_after[1]}")
            break
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    if collision_detected and velocities_reversed:
        print("✓✓✓ SUCCESS: Collision detected and resolved correctly!")
        print("    - Collision was detected")
        print("    - Velocities were reversed")
        print("    - Balls separated after collision")
        return True
    elif collision_detected and not velocities_reversed:
        print("✗✗✗ FAILURE: Collision detected but velocities NOT reversed")
        print("    This means collision response is not working!")
        return False
    else:
        print("✗✗✗ FAILURE: No collision detected (tunneling occurred)")
        return False

if __name__ == "__main__":
    success = test_head_on_collision()
    sys.exit(0 if success else 1)
