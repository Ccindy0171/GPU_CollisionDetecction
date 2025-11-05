#!/usr/bin/env python3
"""
Test 4: Multiple balls falling and colliding with each other (with gravity)
This simulates the actual gravity_fall scenario
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_multi_ball_falling():
    """Test multiple balls falling and colliding"""
    print("=" * 70)
    print("TEST 4: MULTI-BALL FALLING (10 balls, WITH gravity)")
    print("=" * 70)
    
    num_balls = 10
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, 0, -10), (10, 20, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)
    )
    
    # Create balls at different heights, some overlapping horizontally
    # They should collide as they fall
    positions = np.array([
        [0.0, 10.0, 0.0],    # Ball 0
        [0.5, 9.5, 0.0],     # Ball 1 - close to ball 0
        [1.0, 9.0, 0.0],     # Ball 2
        [-0.5, 8.5, 0.0],    # Ball 3 - close to ball 0
        [0.0, 8.0, 0.0],     # Ball 4 - directly below ball 0
        [1.5, 7.5, 0.0],     # Ball 5
        [-1.0, 7.0, 0.0],    # Ball 6
        [0.5, 6.5, 0.0],     # Ball 7
        [-1.5, 6.0, 0.0],    # Ball 8
        [0.0, 5.5, 0.0],     # Ball 9
    ], dtype=np.float32)
    
    velocities = np.zeros((num_balls, 3), dtype=np.float32)
    radii = np.full(num_balls, 0.3, dtype=np.float32)
    masses = np.full(num_balls, 1.0, dtype=np.float32)
    restitutions = np.full(num_balls, 0.7, dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    print(f"\nINITIAL STATE: {num_balls} balls at heights 5.5m to 10m")
    
    # Check initial distances
    print("\nInitial distances between nearby balls:")
    for i in range(num_balls):
        for j in range(i+1, num_balls):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist < 2.0:
                print(f"  Ball {i} <-> Ball {j}: {dist:.3f}m")
    
    total_collisions = 0
    collision_frames = []
    frames_with_collisions = 0
    max_collisions_per_frame = 0
    
    # Track which pairs have collided
    collided_pairs = set()
    
    print("\nSIMULATING (200 frames)...")
    for frame in range(200):
        stats = sim.step()
        
        num_col = stats['num_collisions']
        total_collisions += num_col
        
        if num_col > 0:
            frames_with_collisions += 1
            collision_frames.append(frame)
            max_collisions_per_frame = max(max_collisions_per_frame, num_col)
            
            # Try to identify which balls collided (approximately)
            pos = cp.asnumpy(sim.bodies.positions)
            for i in range(num_balls):
                for j in range(i+1, num_balls):
                    dist = np.linalg.norm(pos[j] - pos[i])
                    if dist <= 0.65:  # Close to touching
                        pair = (min(i,j), max(i,j))
                        if pair not in collided_pairs:
                            collided_pairs.add(pair)
                            if len(collided_pairs) <= 5:  # Only print first 5
                                print(f"  Frame {frame}: Ball {i} <-> Ball {j} collided (dist={dist:.3f}m)")
        
        # Print progress
        if frame % 40 == 0 or (num_col > 0 and frame < 100):
            pos = cp.asnumpy(sim.bodies.positions)
            vel = cp.asnumpy(sim.bodies.velocities)
            min_height = pos[:, 1].min()
            max_height = pos[:, 1].max()
            avg_vel_y = vel[:, 1].mean()
            print(f"  Frame {frame:3d}: collisions={num_col:2d}, "
                  f"height_range=[{min_height:.1f}, {max_height:.1f}], "
                  f"avg_vy={avg_vel_y:6.2f}")
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"  Total collisions: {total_collisions}")
    print(f"  Frames with collisions: {frames_with_collisions}/200")
    print(f"  Max collisions in one frame: {max_collisions_per_frame}")
    print(f"  Unique pairs that collided: {len(collided_pairs)}")
    print(f"  First 10 collision frames: {collision_frames[:10]}")
    
    # Success criteria
    success = total_collisions >= 5 and len(collided_pairs) >= 3
    
    if success:
        print("\n✓✓✓ SUCCESS: Multiple balls collided during fall!")
        print(f"    - {total_collisions} total collisions detected")
        print(f"    - {len(collided_pairs)} different ball pairs collided")
        print(f"    - Collisions happened over {frames_with_collisions} frames")
    else:
        print("\n✗✗✗ FAILURE: Not enough collisions!")
        if total_collisions < 5:
            print(f"    - Only {total_collisions} collisions (expected >= 5)")
        if len(collided_pairs) < 3:
            print(f"    - Only {len(collided_pairs)} pairs collided (expected >= 3)")
    
    return success

if __name__ == "__main__":
    success = test_multi_ball_falling()
    sys.exit(0 if success else 1)
