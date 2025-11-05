#!/usr/bin/env python3
"""
Test 3: Falling ball hitting stationary ball (with gravity)
Ball 0 falls from height and hits Ball 1 on the ground
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_falling_ball_collision():
    """Test falling ball hitting stationary ball"""
    print("=" * 70)
    print("TEST 3: FALLING BALL COLLISION (2 balls, WITH gravity)")
    print("=" * 70)
    
    sim = PhysicsSimulator(
        num_objects=2,
        world_bounds=((-10, 0, -10), (10, 20, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)  # Gravity in -Y direction
    )
    
    # Ball 0: at height 5m, will fall
    # Ball 1: on ground (y=0.3 = radius), stationary
    positions = np.array([
        [0.0, 5.0, 0.0],   # Ball 0 at 5m height
        [0.0, 0.3, 0.0],   # Ball 1 on ground
    ], dtype=np.float32)
    
    velocities = np.array([
        [0.0, 0.0, 0.0],   # Ball 0 stationary initially
        [0.0, 0.0, 0.0],   # Ball 1 stationary
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitutions = np.array([0.8, 0.8], dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    print("\nINITIAL STATE:")
    print(f"  Ball 0: pos={positions[0]}, vel={velocities[0]} (FALLING)")
    print(f"  Ball 1: pos={positions[1]}, vel={velocities[1]} (ON GROUND)")
    print(f"  Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    
    collision_detected = False
    ball0_bounced = False
    ball1_bounced = False
    max_vel_0 = 0
    
    print("\nSIMULATING...")
    for frame in range(200):
        vel_before = cp.asnumpy(sim.bodies.velocities)
        pos_before = cp.asnumpy(sim.bodies.positions)
        
        stats = sim.step()
        
        pos_after = cp.asnumpy(sim.bodies.positions)
        vel_after = cp.asnumpy(sim.bodies.velocities)
        
        # Track max velocity of falling ball
        if abs(vel_after[0, 1]) > max_vel_0:
            max_vel_0 = abs(vel_after[0, 1])
        
        # Check if balls collided
        if stats['num_collisions'] > 0 and not collision_detected:
            dist = np.linalg.norm(pos_after[1] - pos_after[0])
            if dist < 0.65:  # Close to touching
                collision_detected = True
                print(f"\n  Frame {frame}: ✓ COLLISION DETECTED")
                print(f"    Ball 0 pos: {pos_after[0]}, vel: {vel_after[0]}")
                print(f"    Ball 1 pos: {pos_after[1]}, vel: {vel_after[1]}")
                print(f"    Distance: {dist:.4f}m")
                print(f"    Ball 0 had fallen and reached max velocity: {max_vel_0:.2f} m/s")
        
        # Check if ball 0 bounced back up
        if collision_detected and vel_after[0, 1] > 0.5 and not ball0_bounced:
            ball0_bounced = True
            print(f"\n  Frame {frame}: ✓ Ball 0 bounced back (vy={vel_after[0, 1]:.2f} m/s)")
        
        # Check if ball 1 moved
        if collision_detected and abs(vel_after[1, 1]) > 0.5 and not ball1_bounced:
            ball1_bounced = True
            print(f"\n  Frame {frame}: ✓ Ball 1 got hit and moved (vy={vel_after[1, 1]:.2f} m/s)")
        
        # Progress updates
        if frame % 30 == 0:
            print(f"  Frame {frame}: collisions={stats['num_collisions']:3d}, "
                  f"pos0_y={pos_after[0,1]:.2f}, vel0_y={vel_after[0,1]:6.2f}, "
                  f"pos1_y={pos_after[1,1]:.2f}, vel1_y={vel_after[1,1]:6.2f}")
        
        # If both balls bounced, test is complete
        if ball0_bounced and ball1_bounced:
            print(f"\n  Frame {frame}: ✓ Both balls responded to collision")
            break
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    if collision_detected and (ball0_bounced or ball1_bounced):
        print("✓✓✓ SUCCESS: Falling ball collision worked!")
        print(f"    - Collision detected: {collision_detected}")
        print(f"    - Ball 0 bounced back: {ball0_bounced}")
        print(f"    - Ball 1 moved/bounced: {ball1_bounced}")
        print(f"    - Max falling velocity: {max_vel_0:.2f} m/s")
        return True
    else:
        print("✗✗✗ FAILURE:")
        if not collision_detected:
            print("    - No collision detected")
        if not ball0_bounced:
            print("    - Ball 0 did NOT bounce back")
        if not ball1_bounced:
            print("    - Ball 1 did NOT move")
        return False

if __name__ == "__main__":
    success = test_falling_ball_collision()
    sys.exit(0 if success else 1)
