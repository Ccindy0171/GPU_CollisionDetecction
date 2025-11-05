#!/usr/bin/env python3
"""Detailed diagnosis of collision system"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_penetration_depth():
    """Test if penetration is being resolved"""
    print("=" * 70)
    print("PENETRATION DEPTH TEST")
    print("=" * 70)
    
    # Create 10 balls in a tight cluster - will definitely collide
    num_balls = 10
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=1.5,
        dt=1.0/120.0,  # Smaller timestep for accuracy
        gravity=(0, -9.81, 0)
    )
    
    # Place balls in a tight vertical stack
    positions = np.zeros((num_balls, 3), dtype=np.float32)
    for i in range(num_balls):
        positions[i] = [0.0, 5.0 + i * 0.5, 0.0]  # Stacked vertically, overlapping
    
    velocities = np.zeros((num_balls, 3), dtype=np.float32)
    radii = np.full(num_balls, 0.3, dtype=np.float32)
    masses = np.full(num_balls, 1.0, dtype=np.float32)
    restitution = np.full(num_balls, 0.8, dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    print("\n1. INITIAL SETUP:")
    print(f"   {num_balls} balls stacked vertically")
    print(f"   Ball radius: {radii[0]:.2f}m")
    print(f"   Spacing: 0.5m (overlap: {2*radii[0] - 0.5:.2f}m)")
    
    # Check initial overlaps
    def count_overlaps(pos, rad):
        count = 0
        max_penetration = 0
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dist = np.linalg.norm(pos[j] - pos[i])
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    count += 1
                    penetration = min_dist - dist
                    max_penetration = max(max_penetration, penetration)
        return count, max_penetration
    
    overlaps, max_pen = count_overlaps(positions, radii)
    print(f"   Initial overlaps: {overlaps}")
    print(f"   Max penetration: {max_pen:.3f}m")
    
    print("\n2. RUNNING SIMULATION:")
    print(f"   Timestep: {sim.dt:.4f}s (1/{1/sim.dt:.0f} fps)")
    print()
    
    total_collisions_detected = 0
    
    for frame in range(60):
        stats = sim.step()
        pos = cp.asnumpy(sim.bodies.positions)
        vel = cp.asnumpy(sim.bodies.velocities)
        
        overlaps, max_pen = count_overlaps(pos, radii)
        total_collisions_detected += stats['num_collisions']
        
        if frame % 10 == 0 or frame < 5:
            print(f"   Frame {frame:3d}: "
                  f"Detected={stats['num_collisions']:3d}, "
                  f"Overlaps={overlaps:3d}, "
                  f"MaxPen={max_pen:.3f}m, "
                  f"AvgVelY={np.mean(vel[:, 1]):6.2f}")
    
    print(f"\n   Total collision events: {total_collisions_detected}")
    
    print("\n3. FINAL STATE:")
    pos_final = cp.asnumpy(sim.bodies.positions)
    vel_final = cp.asnumpy(sim.bodies.velocities)
    overlaps_final, max_pen_final = count_overlaps(pos_final, radii)
    
    print(f"   Final overlaps: {overlaps_final}")
    print(f"   Final max penetration: {max_pen_final:.3f}m")
    print(f"   Average Y velocity: {np.mean(vel_final[:, 1]):.2f} m/s")
    
    # Check ball positions
    print(f"\n   Ball positions (Y coordinate):")
    for i in range(min(5, num_balls)):
        print(f"      Ball {i}: Y = {pos_final[i, 1]:.3f}m, VelY = {vel_final[i, 1]:.3f} m/s")
    
    print("\n4. ANALYSIS:")
    if overlaps_final > overlaps * 0.5:
        print(f"   ✗ PROBLEM: Still {overlaps_final} overlaps (started with {overlaps})")
        print(f"   → Collision resolution is not working effectively")
    elif total_collisions_detected < overlaps * 5:
        print(f"   ✗ PROBLEM: Only {total_collisions_detected} collisions detected")
        print(f"   → Expected many more for {overlaps} initial overlaps")
    else:
        print(f"   ✓ Collision detection working: {total_collisions_detected} events")
    
    if max_pen_final > 0.1:
        print(f"   ✗ PROBLEM: Max penetration {max_pen_final:.3f}m too large")
        print(f"   → Position correction not strong enough")
    else:
        print(f"   ✓ Penetration controlled: {max_pen_final:.3f}m")
    
    return overlaps_final == 0 and total_collisions_detected > 0

if __name__ == "__main__":
    success = test_penetration_depth()
    sys.exit(0 if success else 1)
