#!/usr/bin/env python3
"""Debug collision response - verify impulses are applied"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp
import numpy as np
from src.simulator import PhysicsSimulator

def test_collision_response():
    """Test if collision response actually changes velocities"""
    print("=" * 60)
    print("Collision Response Debug Test")
    print("=" * 60)
    
    # Create 2 balls moving towards each other
    num_balls = 2
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=((-10, -10, -10), (10, 10, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # No gravity to isolate collision effects
    )
    
    # Ball 1: moving right (+x)
    # Ball 2: moving left (-x)
    # Place them overlapping initially to trigger collision detection
    positions = np.array([
        [-0.2, 0.0, 0.0],   # Ball 1
        [0.2, 0.0, 0.0],    # Ball 2 (distance 0.4m < sum of radii 0.6m)
    ], dtype=np.float32)
    
    velocities = np.array([
        [5.0, 0.0, 0.0],    # Ball 1: moving right
        [-5.0, 0.0, 0.0],   # Ball 2: moving left
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitution = np.array([1.0, 1.0], dtype=np.float32)  # Perfectly elastic
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    print("\n1. INITIAL STATE:")
    print(f"   Ball 1: pos={positions[0]}, vel={velocities[0]}")
    print(f"   Ball 2: pos={positions[1]}, vel={velocities[1]}")
    print(f"   Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"   Sum of radii: {radii[0] + radii[1]:.3f}m")
    print(f"   Currently overlapping: {np.linalg.norm(positions[1] - positions[0]) < radii[0] + radii[1]}")
    
    # Build grid
    sim.build_grid()
    
    print("\n2. COLLISION DETECTION:")
    num_collisions = sim.detect_collisions()
    print(f"   Collisions detected: {num_collisions}")
    
    if num_collisions > 0:
        print(f"   Collision pairs:")
        collision_pairs = cp.asnumpy(sim.collision_pairs[:num_collisions])
        for i, (idx1, idx2) in enumerate(collision_pairs):
            print(f"      Pair {i}: Ball {idx1} <-> Ball {idx2}")
    
    # Get velocities BEFORE collision response
    vel_before = cp.asnumpy(sim.bodies.velocities)
    pos_before = cp.asnumpy(sim.bodies.positions)
    
    print("\n3. BEFORE COLLISION RESPONSE:")
    print(f"   Ball 1: vel={vel_before[0]}")
    print(f"   Ball 2: vel={vel_before[1]}")
    
    # Apply collision response
    if num_collisions > 0:
        sim.resolve_collisions(num_collisions)
        cp.cuda.Stream.null.synchronize()  # Ensure kernel completes
    
    # Get velocities AFTER collision response
    vel_after = cp.asnumpy(sim.bodies.velocities)
    pos_after = cp.asnumpy(sim.bodies.positions)
    
    print("\n4. AFTER COLLISION RESPONSE:")
    print(f"   Ball 1: vel={vel_after[0]}")
    print(f"   Ball 2: vel={vel_after[1]}")
    
    print("\n5. VELOCITY CHANGES:")
    vel_change_1 = vel_after[0] - vel_before[0]
    vel_change_2 = vel_after[1] - vel_before[1]
    print(f"   Ball 1: Δv={vel_change_1}")
    print(f"   Ball 2: Δv={vel_change_2}")
    
    print("\n6. POSITION CHANGES:")
    pos_change_1 = pos_after[0] - pos_before[0]
    pos_change_2 = pos_after[1] - pos_before[1]
    print(f"   Ball 1: Δpos={pos_change_1}")
    print(f"   Ball 2: Δpos={pos_change_2}")
    
    print("\n7. VERIFICATION:")
    # Check if velocities changed
    vel_changed = not np.allclose(vel_before, vel_after, atol=1e-6)
    print(f"   Velocities changed: {vel_changed}")
    
    # Check if velocities reversed (elastic collision)
    # Ball 1 should now be moving left (negative x)
    # Ball 2 should now be moving right (positive x)
    ball1_reversed = vel_after[0, 0] < 0
    ball2_reversed = vel_after[1, 0] > 0
    print(f"   Ball 1 velocity reversed (should be negative): {vel_after[0, 0]:.3f} {'✓' if ball1_reversed else '✗'}")
    print(f"   Ball 2 velocity reversed (should be positive): {vel_after[1, 0]:.3f} {'✓' if ball2_reversed else '✗'}")
    
    # Check momentum conservation (should be zero for equal masses and opposite velocities)
    momentum_before = vel_before[0] + vel_before[1]
    momentum_after = vel_after[0] + vel_after[1]
    print(f"   Momentum before: {momentum_before}")
    print(f"   Momentum after: {momentum_after}")
    print(f"   Momentum conserved: {np.allclose(momentum_before, momentum_after, atol=1e-3)}")
    
    # Check energy conservation (elastic collision)
    ke_before = 0.5 * (np.sum(vel_before[0]**2) + np.sum(vel_before[1]**2))
    ke_after = 0.5 * (np.sum(vel_after[0]**2) + np.sum(vel_after[1]**2))
    print(f"   KE before: {ke_before:.3f} J")
    print(f"   KE after: {ke_after:.3f} J")
    print(f"   Energy conserved: {np.allclose(ke_before, ke_after, atol=0.1)}")
    
    if vel_changed and ball1_reversed and ball2_reversed:
        print("\n✓ SUCCESS: Collision response is working correctly!")
    else:
        print("\n✗ FAILURE: Collision response is NOT working!")
        
    return vel_changed and ball1_reversed and ball2_reversed

if __name__ == "__main__":
    success = test_collision_response()
    sys.exit(0 if success else 1)
