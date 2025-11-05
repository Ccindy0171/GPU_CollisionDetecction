#!/usr/bin/env python3
"""
Diagnostic test for tunneling/pass-through issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
from src.init_helper import generate_non_overlapping_positions

def main():
    print("=" * 70)
    print("TUNNELING DIAGNOSTIC TEST")
    print("=" * 70)
    
    NUM_OBJECTS = 10
    WORLD_BOUNDS = ((-5, 0, -5), (5, 10, 5))
    
    # Test with different cell sizes
    for cell_size in [2.0, 1.0, 0.8]:
        print(f"\n{'=' * 70}")
        print(f"Testing with CELL_SIZE = {cell_size}")
        print(f"{'=' * 70}")
        
        sim = PhysicsSimulator(
            num_objects=NUM_OBJECTS,
            world_bounds=WORLD_BOUNDS,
            cell_size=cell_size,
            device_id=0,
            dt=1.0 / 120.0,  # Small timestep
            gravity=(0.0, -9.81, 0.0),
            damping=0.01
        )
        
        # Generate scene
        np.random.seed(42)
        radii = np.random.lognormal(mean=-1.0, sigma=0.5, size=NUM_OBJECTS)
        radii = np.clip(radii, 0.15, 0.8).astype(np.float32)
        
        positions = generate_non_overlapping_positions(
            num_objects=NUM_OBJECTS,
            radii=radii,
            world_bounds=WORLD_BOUNDS,
            max_attempts=50
        )
        
        velocities = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
        for i in range(NUM_OBJECTS):
            if np.random.random() < 0.5:
                velocities[i, 1] = np.random.uniform(-3, -1)
        
        masses = np.ones(NUM_OBJECTS, dtype=np.float32)
        restitutions = np.full(NUM_OBJECTS, 0.7, dtype=np.float32)
        
        sim.bodies.positions[:] = cp.asarray(positions)
        sim.bodies.velocities[:] = cp.asarray(velocities)
        sim.bodies.radii[:] = cp.asarray(radii)
        sim.bodies.masses[:] = cp.asarray(masses)
        sim.bodies.restitutions[:] = cp.asarray(restitutions)
        
        print(f"\nInitial setup:")
        print(f"  Balls: {NUM_OBJECTS}")
        print(f"  Radius range: {radii.min():.3f} - {radii.max():.3f}m")
        print(f"  Cell size: {cell_size}m")
        print(f"  Max radius / cell size ratio: {radii.max() / cell_size:.2f}")
        
        # Run simulation
        total_collisions = 0
        max_overlaps = 0
        frames_with_overlaps = 0
        
        print(f"\nRunning 300 frames (5 seconds)...")
        for frame in range(300):
            # Run 2 steps per frame
            stats1 = sim.step()
            stats2 = sim.step()
            total_collisions += stats1['num_collisions'] + stats2['num_collisions']
            
            # Check overlaps
            pos = cp.asnumpy(sim.bodies.positions)
            num_overlaps = 0
            max_pen = 0.0
            
            for i in range(NUM_OBJECTS):
                for j in range(i+1, NUM_OBJECTS):
                    dist = np.linalg.norm(pos[j] - pos[i])
                    required = radii[i] + radii[j]
                    if dist < required - 1e-6:
                        pen = required - dist
                        num_overlaps += 1
                        max_pen = max(max_pen, pen)
            
            if num_overlaps > 0:
                frames_with_overlaps += 1
                max_overlaps = max(max_overlaps, num_overlaps)
                if frame % 30 == 0 or num_overlaps > 5:
                    print(f"  Frame {frame:3d}: {num_overlaps} overlaps, max_pen={max_pen:.4f}m, "
                          f"coll={stats1['num_collisions']}+{stats2['num_collisions']}")
        
        print(f"\nResults:")
        print(f"  Total collisions: {total_collisions}")
        print(f"  Frames with overlaps: {frames_with_overlaps}/300")
        print(f"  Max simultaneous overlaps: {max_overlaps}")
        
        if frames_with_overlaps == 0:
            print(f"  ✓✓✓ NO TUNNELING DETECTED!")
        elif frames_with_overlaps < 30:
            print(f"  ⚠ Minor tunneling (< 10% of frames)")
        else:
            print(f"  ✗✗✗ SIGNIFICANT TUNNELING!")

if __name__ == "__main__":
    main()
