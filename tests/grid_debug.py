#!/usr/bin/env python3
"""Debug grid construction and collision detection"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_grid_construction():
    """Test if grid is constructed correctly for 2 overlapping balls"""
    print("=" * 60)
    print("Grid Construction Debug Test")
    print("=" * 60)
    
    # Create simulator
    num_balls = 2
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)
    )
    
    # Set ball properties
    positions = np.array([
        [0.0, 5.0, 0.0],   # Ball 1
        [0.5, 5.0, 0.0],   # Ball 2 (distance 0.5m)
    ], dtype=np.float32)
    
    velocities = np.zeros((num_balls, 3), dtype=np.float32)
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitution = np.array([0.8, 0.8], dtype=np.float32)
    
    # Initialize ball data
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    print(f"\n1. INITIAL STATE:")
    print(f"   Num balls: {num_balls}")
    print(f"   Ball 1 pos: {positions[0]}")
    print(f"   Ball 2 pos: {positions[1]}")
    print(f"   Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"   Sum of radii: {radii[0] + radii[1]:.3f}m")
    print(f"   Should overlap: {np.linalg.norm(positions[1] - positions[0]) < radii[0] + radii[1]}")
    
    print(f"\n2. GRID PROPERTIES:")
    print(f"   World min: {cp.asnumpy(sim.grid.world_min)}")
    print(f"   World max: {cp.asnumpy(sim.grid.world_max)}")
    print(f"   Cell size: {float(sim.grid.cell_size):.3f}m")
    print(f"   Resolution: {cp.asnumpy(sim.grid.resolution)}")
    print(f"   Total cells: {sim.grid.total_cells}")
    
    # Build spatial structure
    sim.build_grid()
    
    print(f"\n3. GRID HASHES:")
    grid_hashes = cp.asnumpy(sim.grid_hashes)
    print(f"   Ball 1 hash: {grid_hashes[0]}")
    print(f"   Ball 2 hash: {grid_hashes[1]}")
    print(f"   Same cell: {grid_hashes[0] == grid_hashes[1]}")
    
    # Convert hashes to grid coordinates
    def hash_to_coord(hash_val, resolution):
        res_x, res_y, res_z = resolution
        z = hash_val // (res_y * res_x)
        remainder = hash_val % (res_y * res_x)
        y = remainder // res_x
        x = remainder % res_x
        return (x, y, z)
    
    resolution = cp.asnumpy(sim.grid.resolution)
    coord1 = hash_to_coord(grid_hashes[0], resolution)
    coord2 = hash_to_coord(grid_hashes[1], resolution)
    print(f"   Ball 1 grid coord: {coord1}")
    print(f"   Ball 2 grid coord: {coord2}")
    
    print(f"\n4. SORTED INDICES:")
    sorted_indices = cp.asnumpy(sim.sorted_indices)
    print(f"   Sorted indices: {sorted_indices}")
    
    sorted_hashes = cp.asnumpy(sim.grid_hashes[sim.sorted_indices])
    print(f"   Sorted hashes: {sorted_hashes}")
    
    print(f"\n5. CELL START/END ARRAYS:")
    cell_starts = cp.asnumpy(sim.grid.cell_starts)
    cell_ends = cp.asnumpy(sim.grid.cell_ends)
    
    # Find non-empty cells
    non_empty = np.where(cell_starts >= 0)[0]
    print(f"   Non-empty cells: {len(non_empty)}")
    
    for cell_idx in non_empty[:10]:  # Show first 10 non-empty cells
        start = cell_starts[cell_idx]
        end = cell_ends[cell_idx]
        coord = hash_to_coord(cell_idx, resolution)
        print(f"   Cell {cell_idx} (coord {coord}): start={start}, end={end}, count={end-start}")
        
        # Show which balls are in this cell
        for i in range(start, end):
            ball_idx = sorted_indices[i]
            ball_pos = positions[ball_idx]
            print(f"      -> Ball {ball_idx} at {ball_pos}")
    
    print(f"\n6. COLLISION DETECTION:")
    num_collisions = sim.detect_collisions()
    print(f"   Collisions detected: {num_collisions}")
    
    if num_collisions > 0:
        print(f"   ✓ SUCCESS: Collision detected!")
        # Show collision pairs
        collision_pairs = cp.asnumpy(sim.collision_pairs[:num_collisions])
        for i, (idx1, idx2) in enumerate(collision_pairs):
            pos1 = positions[idx1]
            pos2 = positions[idx2]
            dist = np.linalg.norm(pos2 - pos1)
            sum_r = radii[idx1] + radii[idx2]
            print(f"   Pair {i}: Ball {idx1} <-> Ball {idx2}, dist={dist:.3f}, sum_r={sum_r:.3f}")
    else:
        print(f"   ✗ FAILURE: No collisions detected!")
        
        # Debug: Check if balls are in adjacent cells
        print(f"\n7. DEBUGGING - Cell Neighbors:")
        for hash_val in [grid_hashes[0], grid_hashes[1]]:
            coord = hash_to_coord(hash_val, resolution)
            print(f"   Cell {hash_val} (coord {coord}) neighbors:")
            
            # Check 3x3x3 neighborhood
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny, nz = coord[0] + dx, coord[1] + dy, coord[2] + dz
                        
                        # Check bounds
                        if (0 <= nx < resolution[0] and 
                            0 <= ny < resolution[1] and 
                            0 <= nz < resolution[2]):
                            
                            neighbor_hash = nz * resolution[1] * resolution[0] + ny * resolution[0] + nx
                            start = cell_starts[neighbor_hash]
                            end = cell_ends[neighbor_hash]
                            
                            if start >= 0:
                                count = end - start
                                print(f"      Neighbor ({nx},{ny},{nz}) hash={neighbor_hash}: {count} balls")

if __name__ == "__main__":
    test_grid_construction()
