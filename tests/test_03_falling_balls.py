#!/usr/bin/env python3
"""
Test 03: Multiple Balls Falling with Gravity
10 balls falling and colliding
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator

def main():
    print("=" * 70)
    print("FALLING BALLS TEST - 10 Balls with Gravity")
    print("=" * 70)
    
    # Create simulator with 10 balls, WITH GRAVITY
    num_objects = 10
    sim = PhysicsSimulator(
        num_objects=num_objects,
        world_bounds=((-5, 0, -5), (5, 10, 5)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)  # Normal gravity
    )
    
    print("\nTest Setup:")
    print("  10 balls arranged in a grid above ground")
    print("  All balls start at rest")
    print("  Gravity: -9.81 m/s² (downward)")
    print("  Expected: Balls fall, collide with each other and ground")
    print("            Should form a pile without tunneling")
    
    # Initialize balls - arranged in a 2x5 grid
    positions = np.zeros((num_objects, 3), dtype=np.float32)
    for i in range(num_objects):
        row = i // 5
        col = i % 5
        positions[i] = [
            col * 0.8 - 1.6,  # Spread across X
            6.0 + row * 0.8,  # Two layers in Y
            0.0               # Center in Z
        ]
    
    velocities = np.zeros((num_objects, 3), dtype=np.float32)
    
    radii = np.full(num_objects, 0.3, dtype=np.float32)
    masses = np.full(num_objects, 1.0, dtype=np.float32)
    restitutions = np.full(num_objects, 0.7, dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    # Colors: rainbow
    
    print(f"\nInitial positions:")
    for i in range(num_objects):
        print(f"  Ball {i}: {positions[i]}")
    
    # Create visualizer
    world_bounds = ((-5, 0, -5), (5, 10, 5))
    
    # Create video
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'falling_balls_test.mp4')
    
    # Run simulation
    total_frames = 300  # 5 seconds
    
    print("\nRunning simulation...")
    
    max_penetration_ever = 0.0
    collision_count_history = []
    
    for frame in range(total_frames):
        # Step physics
        stats = sim.step()
        
        # Get current state
        pos = cp.asnumpy(sim.bodies.positions)
        vel = cp.asnumpy(sim.bodies.velocities)
        
        # Calculate overlaps between balls (not ground)
        max_penetration = 0.0
        num_overlaps = 0
        
        for i in range(num_objects):
            for j in range(i+1, num_objects):
                dist = np.linalg.norm(pos[j] - pos[i])
                min_dist = radii[i] + radii[j]
                overlap = max(0, min_dist - dist)
                if overlap > 0:
                    max_penetration = max(max_penetration, overlap)
                    num_overlaps += 1
        
        if max_penetration > max_penetration_ever:
            max_penetration_ever = max_penetration
        
        collision_count_history.append(stats['num_collisions'])
        
        # Check lowest ball height
        min_y = np.min(pos[:, 1])
        max_y = np.max(pos[:, 1])
        
        # Update visualization
        info = f"Frame: {frame}/{total_frames}\n"
        info += f"Time: {frame/60:.2f}s\n"
        info += f"Collisions: {stats['num_collisions']}\n"
        info += f"Overlaps: {num_overlaps}\n"
        if max_penetration > 0:
            info += f"Max penetration: {max_penetration:.4f}m\n"
        info += f"Height range: {min_y:.2f} - {max_y:.2f}m"
        
        
        # Print progress
        if frame % 30 == 0:
            print(f"Frame {frame:3d}: coll={stats['num_collisions']:3d}, "
                  f"overlaps={num_overlaps}, pen={max_penetration:.4f}m, "
                  f"height={min_y:.2f}-{max_y:.2f}m")
    
    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    pos_final = cp.asnumpy(sim.bodies.positions)
    
    # Check final overlaps
    final_overlaps = 0
    final_max_pen = 0.0
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            dist = np.linalg.norm(pos_final[j] - pos_final[i])
            min_dist = radii[i] + radii[j]
            overlap = max(0, min_dist - dist)
            if overlap > 0:
                final_overlaps += 1
                final_max_pen = max(final_max_pen, overlap)
    
    print(f"Max penetration during simulation: {max_penetration_ever:.4f}m")
    print(f"Final overlapping pairs: {final_overlaps}")
    if final_overlaps > 0:
        print(f"Final max penetration: {final_max_pen:.4f}m")
    
    # Collision statistics
    total_collisions = sum(collision_count_history)
    avg_collisions = np.mean(collision_count_history)
    max_collisions = np.max(collision_count_history)
    
    print(f"\nCollision statistics:")
    print(f"  Total collisions: {total_collisions}")
    print(f"  Average per frame: {avg_collisions:.1f}")
    print(f"  Maximum in one frame: {max_collisions}")
    
    # Check if balls settled
    final_min_y = np.min(pos_final[:, 1])
    final_max_y = np.max(pos_final[:, 1])
    print(f"\nFinal height range: {final_min_y:.2f} - {final_max_y:.2f}m")
    
    # Check velocities
    vel_final = cp.asnumpy(sim.bodies.velocities)
    vel_magnitudes = np.linalg.norm(vel_final, axis=1)
    avg_vel = np.mean(vel_magnitudes)
    max_vel = np.max(vel_magnitudes)
    print(f"\nFinal velocities:")
    print(f"  Average magnitude: {avg_vel:.2f} m/s")
    print(f"  Maximum magnitude: {max_vel:.2f} m/s")
    
    # Final verdict
    print("\n" + "=" * 70)
    if final_overlaps == 0 and max_penetration_ever < 0.1:
        print("✓✓✓ TEST PASSED ✓✓✓")
        print("No tunneling detected!")
        print("Collision physics working correctly!")
    elif final_overlaps == 0 and max_penetration_ever < 0.2:
        print("✓ TEST MOSTLY PASSED")
        print(f"Max penetration {max_penetration_ever:.4f}m is acceptable")
        print("No final overlaps")
    else:
        print("✗ TEST FAILED")
        if final_overlaps > 0:
            print(f"  - {final_overlaps} overlapping pairs remain")
        if max_penetration_ever >= 0.2:
            print(f"  - Excessive penetration: {max_penetration_ever:.4f}m")
    print("=" * 70)
    
    # Close
    

if __name__ == "__main__":
    main()
