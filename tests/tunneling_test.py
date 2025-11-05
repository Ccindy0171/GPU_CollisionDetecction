#!/usr/bin/env python3
"""Test high-speed collision to verify no tunneling"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_high_speed_collision():
    """Test if fast-moving balls tunnel through each other"""
    print("=" * 60)
    print("High-Speed Collision Test (Anti-Tunneling)")
    print("=" * 60)
    
    # Create 2 balls moving VERY fast towards each other
    num_balls = 2
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # No gravity
    )
    
    # Balls far apart, moving EXTREMELY fast towards each other
    # At 100 m/s, they travel 1.67m per frame at 60fps
    # This is much more than their combined diameter (0.6m)
    positions = np.array([
        [-5.0, 0.0, 0.0],   # Ball 1 on left
        [5.0, 0.0, 0.0],    # Ball 2 on right
    ], dtype=np.float32)
    
    velocities = np.array([
        [100.0, 0.0, 0.0],    # Ball 1: moving right at 100 m/s
        [-100.0, 0.0, 0.0],   # Ball 2: moving left at 100 m/s
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitution = np.array([0.9, 0.9], dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    print("\n1. INITIAL STATE:")
    print(f"   Ball 1: pos={positions[0]}, vel={velocities[0]}")
    print(f"   Ball 2: pos={positions[1]}, vel={velocities[1]}")
    print(f"   Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"   Relative speed: 200 m/s")
    print(f"   Distance traveled per frame (60fps): {200 * (1/60):.3f}m")
    print(f"   Combined diameter: {(radii[0] + radii[1]) * 2:.3f}m")
    print(f"   Can tunnel in one frame: {200 * (1/60) > (radii[0] + radii[1])}")
    
    print("\n2. SIMULATING COLLISION:")
    collision_detected = False
    tunneled = False
    
    for frame in range(200):  # 200 frames = 3.33 seconds
        # Store positions before step
        pos_before = cp.asnumpy(sim.bodies.positions)
        vel_before = cp.asnumpy(sim.bodies.velocities)
        
        # Step simulation
        stats = sim.step()
        
        # Get positions after step
        pos_after = cp.asnumpy(sim.bodies.positions)
        vel_after = cp.asnumpy(sim.bodies.velocities)
        
        # Calculate distance
        dist = np.linalg.norm(pos_after[1] - pos_after[0])
        min_dist = radii[0] + radii[1]
        
        # Check if collision detected
        if stats['num_collisions'] > 0:
            if not collision_detected:
                collision_detected = True
                print(f"\n   Frame {frame}: COLLISION DETECTED")
                print(f"      Distance: {dist:.3f}m")
                print(f"      Ball 1: pos={pos_after[0]}, vel={vel_after[0]}")
                print(f"      Ball 2: pos={pos_after[1]}, vel={vel_after[1]}")
        
        # Check if balls have passed through each other (tunneled)
        # If ball 1 is now to the right of ball 2, they passed through
        if pos_before[0, 0] < pos_before[1, 0]:  # Initially ball 1 is on left
            if pos_after[0, 0] > pos_after[1, 0]:  # Now ball 1 is on right
                if not collision_detected:
                    tunneled = True
                    print(f"\n   Frame {frame}: ✗ TUNNELING DETECTED!")
                    print(f"      Balls passed through without collision")
                    print(f"      Ball 1: {pos_before[0, 0]:.3f} -> {pos_after[0, 0]:.3f}")
                    print(f"      Ball 2: {pos_before[1, 0]:.3f} -> {pos_after[1, 0]:.3f}")
                    break
        
        # If collision detected and balls are separating, test complete
        if collision_detected and dist > min_dist * 1.1:
            print(f"\n   Frame {frame}: Balls separated successfully")
            print(f"      Distance: {dist:.3f}m (> {min_dist * 1.1:.3f}m)")
            print(f"      Ball 1 velocity: {vel_after[0]}")
            print(f"      Ball 2 velocity: {vel_after[1]}")
            break
    
    print("\n3. RESULTS:")
    if tunneled:
        print("   ✗ FAILURE: Balls tunneled through each other!")
        return False
    elif collision_detected:
        print("   ✓ SUCCESS: Collision detected and resolved!")
        # Check if velocities reversed
        vel_final = cp.asnumpy(sim.bodies.velocities)
        ball1_reversed = vel_final[0, 0] < 0  # Should be moving left now
        ball2_reversed = vel_final[1, 0] > 0  # Should be moving right now
        print(f"   Ball 1 velocity reversed: {ball1_reversed}")
        print(f"   Ball 2 velocity reversed: {ball2_reversed}")
        return ball1_reversed and ball2_reversed
    else:
        print("   ⚠ WARNING: No collision detected (balls still approaching?)")
        return False

if __name__ == "__main__":
    success = test_high_speed_collision()
    sys.exit(0 if success else 1)
