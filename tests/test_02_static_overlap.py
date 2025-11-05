#!/usr/bin/env python3
"""
Test 02: Static Overlapping Balls

Place multiple balls in overlapping positions
Check if collision response can separate them

This is a headless physics validation test (no visualization).
Run with: python tests/test_02_static_overlap.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator

def main():
    print("=" * 70)
    print("STATIC OVERLAP TEST - Multiple Balls")
    print("=" * 70)
    
    # Create simulator with 5 balls, NO GRAVITY
    num_objects = 5
    sim = PhysicsSimulator(
        num_objects=num_objects,
        world_bounds=((-5, -5, -5), (5, 5, 5)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # NO GRAVITY
    )
    
    print("\nTest Setup:")
    print("  5 balls placed in overlapping positions")
    print("  All balls start at rest (zero velocity)")
    print("  No gravity - pure collision response test")
    print("  Expected: Balls push apart until no longer overlapping")
    
    # Initialize balls - all at origin, heavily overlapping
    positions = np.array([
        [0.0, 0.0, 0.0],   # Ball 0 at center
        [0.2, 0.0, 0.0],   # Ball 1 slightly right
        [-0.2, 0.0, 0.0],  # Ball 2 slightly left
        [0.0, 0.2, 0.0],   # Ball 3 slightly up
        [0.0, -0.2, 0.0],  # Ball 4 slightly down
    ], dtype=np.float32)
    
    velocities = np.zeros((num_objects, 3), dtype=np.float32)  # All at rest
    
    radii = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    restitutions = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    print(f"\nInitial overlaps:")
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            dist = np.linalg.norm(positions[j] - positions[i])
            min_dist = radii[i] + radii[j]
            overlap = max(0, min_dist - dist)
            if overlap > 0:
                print(f"  Ball {i} <-> Ball {j}: dist={dist:.3f}m, overlap={overlap:.3f}m")
    
    # Run simulation
    total_frames = 300  # 5 seconds
    
    print("\nRunning simulation...")
    
    overlap_history = []
    
    for frame in range(total_frames):
        # Step physics
        stats = sim.step()
        
        # Get current state
        pos = cp.asnumpy(sim.bodies.positions)
        vel = cp.asnumpy(sim.bodies.velocities)
        
        # Calculate overlaps
        total_overlap = 0
        num_overlaps = 0
        max_overlap = 0
        
        for i in range(num_objects):
            for j in range(i+1, num_objects):
                dist = np.linalg.norm(pos[j] - pos[i])
                min_dist = radii[i] + radii[j]
                overlap = max(0, min_dist - dist)
                if overlap > 0:
                    total_overlap += overlap
                    num_overlaps += 1
                    max_overlap = max(max_overlap, overlap)
        
        overlap_history.append((num_overlaps, max_overlap, total_overlap))
        
        # Print progress
        if frame % 30 == 0:
            print(f"Frame {frame:3d}: overlaps={num_overlaps}, "
                  f"max_overlap={max_overlap:.4f}m, "
                  f"collisions={stats['num_collisions']}")
    
    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    final_overlaps = overlap_history[-1][0]
    initial_overlaps = overlap_history[0][0]
    
    print(f"Initial overlapping pairs: {initial_overlaps}")
    print(f"Final overlapping pairs: {final_overlaps}")
    print(f"Initial max overlap: {overlap_history[0][1]:.4f}m")
    print(f"Final max overlap: {overlap_history[-1][1]:.4f}m")
    
    # Check if overlaps were resolved
    print("\nOverlap resolution timeline:")
    for frame in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 299]:
        if frame < len(overlap_history):
            n, m, t = overlap_history[frame]
            print(f"  Frame {frame:3d}: {n} overlaps, max={m:.4f}m, total={t:.4f}m")
    
    # Final verdict
    print("\n" + "=" * 70)
    if final_overlaps == 0:
        print("✓✓✓ TEST PASSED ✓✓✓")
        print("All overlaps successfully resolved!")
    elif final_overlaps < initial_overlaps:
        print("⚠ PARTIAL SUCCESS")
        print(f"Overlaps reduced from {initial_overlaps} to {final_overlaps}")
        print("Some overlaps remain - collision response needs improvement")
    else:
        print("✗✗✗ TEST FAILED ✗✗✗")
        print("Collision response did not resolve overlaps!")
    print("=" * 70)

if __name__ == "__main__":
    main()
