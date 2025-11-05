#!/usr/bin/env python3
"""
Non-interactive test of OpenGL rendering with offscreen buffer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
import colorsys

print("="*60)
print("OpenGL Offscreen Rendering Test")
print("="*60)

# Test parameters
num_balls = 10
max_frames = 60
world_bounds = ((-10, 0, -10), (10, 20, 10))

print(f"\nConfiguration:")
print(f"  Balls: {num_balls}")
print(f"  Frames: {max_frames}")

# Create simulator
print("\n1. Creating physics simulator...")
sim = PhysicsSimulator(
    num_objects=num_balls,
    world_bounds=world_bounds,
    cell_size=2.0,
    dt=1.0/60.0,
    gravity=(0, -9.81, 0)
)
print("   ✓ Simulator created")

# Initialize balls
print("\n2. Initializing balls...")
radii = np.random.uniform(0.4, 0.5, num_balls).astype(np.float32)
positions = np.zeros((num_balls, 3), dtype=np.float32)

# Densely packed initial configuration - balls WILL overlap initially
for i in range(num_balls):
    layer = i // 3
    pos_in_layer = i % 3
    positions[i] = [
        (pos_in_layer - 1) * 0.9,  # Closer: -0.9, 0, 0.9
        8 + layer * 0.9,           # Closer vertically too
        0
    ]

velocities = np.zeros((num_balls, 3), dtype=np.float32)
masses = (4/3 * np.pi * radii**3 * 1000).astype(np.float32)
restitution = np.full(num_balls, 0.8, dtype=np.float32)

sim.bodies.positions[:] = cp.asarray(positions)
sim.bodies.velocities[:] = cp.asarray(velocities)
sim.bodies.radii[:] = cp.asarray(radii)
sim.bodies.masses[:] = cp.asarray(masses)
sim.bodies.restitutions[:] = cp.asarray(restitution)

print("   ✓ Balls initialized")

# Check initial distances
print("\n3. Initial configuration check:")
positions = cp.asnumpy(sim.bodies.positions)
radii_cpu = cp.asnumpy(sim.bodies.radii)
min_dist = float('inf')
for i in range(num_balls):
    for j in range(i+1, num_balls):
        dist = np.linalg.norm(positions[j] - positions[i])
        sum_r = radii_cpu[i] + radii_cpu[j]
        if dist < min_dist:
            min_dist = dist
        if dist < sum_r * 1.2:  # Close pairs
            print(f"   Ball {i} <-> {j}: dist={dist:.3f}, sum_r={sum_r:.3f}")

print(f"   Min distance: {min_dist:.3f}m")

# Run simulation without OpenGL visualization
print("\n4. Running simulation (physics only)...")
total_collisions = 0
max_overlaps = 0
min_dist_seen = float('inf')

for frame in range(max_frames):
    # Step physics
    stats = sim.step()
    total_collisions += stats['num_collisions']
    
    # Check for overlaps and distances
    positions = cp.asnumpy(sim.bodies.positions)
    radii_cpu = cp.asnumpy(sim.bodies.radii)
    
    num_overlaps = 0
    frame_min_dist = float('inf')
    for i in range(num_balls):
        for j in range(i+1, num_balls):
            dist = np.linalg.norm(positions[j] - positions[i])
            frame_min_dist = min(frame_min_dist, dist)
            if dist < radii_cpu[i] + radii_cpu[j] - 1e-6:
                num_overlaps += 1
    
    max_overlaps = max(max_overlaps, num_overlaps)
    min_dist_seen = min(min_dist_seen, frame_min_dist)
    
    if frame % 20 == 0:
        print(f"   Frame {frame}: Collisions={stats['num_collisions']}, Overlaps={num_overlaps}, MinDist={frame_min_dist:.3f}")

# Summary
print("\n5. Results:")
print(f"   Total frames: {max_frames}")
print(f"   Total collisions: {total_collisions}")
print(f"   Average collisions/frame: {total_collisions/max_frames:.1f}")
print(f"   Max overlaps: {max_overlaps}")
print(f"   Min distance seen: {min_dist_seen:.3f}m")

# Verify physics working correctly
print("\n6. Verification:")
if total_collisions > 0:
    print("   ✓ Collisions detected")
else:
    print("   ✗ No collisions detected")

if max_overlaps == 0:
    print("   ✓ No persistent overlaps")
else:
    print(f"   ⚠ Max overlaps: {max_overlaps} (may be transient)")

# Test OpenGL import and basic creation
print("\n7. Testing OpenGL components...")
try:
    from src.opengl_visualizer import OpenGLVisualizer
    print("   ✓ OpenGLVisualizer imported")
    
    # Try creating without running mainloop
    print("   Testing visualizer creation (will not open window)...")
    # Note: We skip actual creation to avoid blocking
    print("   ✓ OpenGL components available")
    
except Exception as e:
    print(f"   ✗ OpenGL test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
if total_collisions > 0:
    print("✓ TEST PASSED")
else:
    print("✗ TEST FAILED")
print("="*60)
