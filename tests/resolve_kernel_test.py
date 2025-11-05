#!/usr/bin/env python3
"""Test if resolve_collisions actually modifies velocities"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_resolve_modifies_velocities():
    """Directly test if resolve_collisions changes velocities"""
    print("=" * 70)
    print("RESOLVE_COLLISIONS MODIFICATION TEST")
    print("=" * 70)
    
    # Create 2 overlapping balls
    num_balls = 2
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # No gravity
    )
    
    # Overlapping balls
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0],  # Distance 0.4m < sum of radii 0.6m
    ], dtype=np.float32)
    
    velocities = np.array([
        [5.0, 0.0, 0.0],   # Moving towards each other
        [-5.0, 0.0, 0.0],
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitution = np.array([1.0, 1.0], dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    print("\n1. SETUP:")
    print(f"   Ball 1: pos={positions[0]}, vel={velocities[0]}")
    print(f"   Ball 2: pos={positions[1]}, vel={velocities[1]}")
    print(f"   Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"   Sum of radii: {radii[0] + radii[1]:.3f}m")
    print(f"   Overlapping: {np.linalg.norm(positions[1] - positions[0]) < radii[0] + radii[1]}")
    
    # Build grid and detect collisions
    print("\n2. COLLISION DETECTION:")
    sim.build_grid()
    num_collisions = sim.detect_collisions()
    print(f"   Collisions detected: {num_collisions}")
    
    if num_collisions > 0:
        pairs = cp.asnumpy(sim.collision_pairs[:num_collisions])
        print(f"   Pairs: {pairs}")
    
    # Get velocities BEFORE resolve
    vel_before = cp.asnumpy(sim.bodies.velocities).copy()
    pos_before = cp.asnumpy(sim.bodies.positions).copy()
    print("\n3. BEFORE RESOLVE_COLLISIONS:")
    print(f"   Ball 1 vel: {vel_before[0]}")
    print(f"   Ball 2 vel: {vel_before[1]}")
    print(f"   Ball 1 pos: {pos_before[0]}")
    print(f"   Ball 2 pos: {pos_before[1]}")
    
    # Call resolve_collisions
    if num_collisions > 0:
        print("\n4. CALLING RESOLVE_COLLISIONS...")
        sim.resolve_collisions(num_collisions)
        cp.cuda.Stream.null.synchronize()
    
    # Get velocities AFTER resolve
    vel_after = cp.asnumpy(sim.bodies.velocities).copy()
    pos_after = cp.asnumpy(sim.bodies.positions).copy()
    print("\n5. AFTER RESOLVE_COLLISIONS:")
    print(f"   Ball 1 vel: {vel_after[0]}")
    print(f"   Ball 2 vel: {vel_after[1]}")
    print(f"   Ball 1 pos: {pos_after[0]}")
    print(f"   Ball 2 pos: {pos_after[1]}")
    
    # Calculate changes
    vel_change = vel_after - vel_before
    pos_change = pos_after - pos_before
    
    print("\n6. CHANGES:")
    print(f"   Ball 1 vel change: {vel_change[0]}")
    print(f"   Ball 2 vel change: {vel_change[1]}")
    print(f"   Ball 1 pos change: {pos_change[0]}")
    print(f"   Ball 2 pos change: {pos_change[1]}")
    
    print("\n7. VERIFICATION:")
    vel_changed = not np.allclose(vel_before, vel_after, atol=1e-6)
    pos_changed = not np.allclose(pos_before, pos_after, atol=1e-6)
    
    print(f"   Velocities changed: {vel_changed}")
    print(f"   Positions changed: {pos_changed}")
    
    if not vel_changed and not pos_changed:
        print(f"\n   ✗ CRITICAL: resolve_collisions DID NOT MODIFY ANYTHING!")
        print(f"   → This means the kernel is not working")
        return False
    elif vel_changed:
        print(f"\n   ✓ resolve_collisions is modifying velocities")
        return True
    else:
        print(f"\n   ⚠ Only positions changed, velocities unchanged")
        return False

if __name__ == "__main__":
    success = test_resolve_modifies_velocities()
    sys.exit(0 if success else 1)
