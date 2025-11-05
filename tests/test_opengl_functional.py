#!/usr/bin/env python3
"""
Functional test of OpenGL visualizer with physics simulation
Tests collision detection and rendering together
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
from src.opengl_visualizer import OpenGLVisualizer
import colorsys

print("="*60)
print("OpenGL + Physics Functional Test")
print("="*60)

# Test parameters
num_balls = 10
max_frames = 120  # 2 seconds
world_bounds = ((-10, 0, -10), (10, 20, 10))

print(f"\nConfiguration:")
print(f"  Balls: {num_balls}")
print(f"  Frames: {max_frames}")
print(f"  World: {world_bounds}")

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
radii = np.random.uniform(0.3, 0.6, num_balls).astype(np.float32)
positions = np.zeros((num_balls, 3), dtype=np.float32)

for i in range(num_balls):
    positions[i] = [
        np.random.uniform(-8, 8),
        10 + i * 2,
        np.random.uniform(-8, 8)
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

# Generate colors
colors = np.zeros((num_balls, 3), dtype=np.float32)
for i in range(num_balls):
    hue = (i * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    colors[i] = rgb

# Create visualizer
print("\n3. Creating OpenGL visualizer...")
try:
    vis = OpenGLVisualizer(
        world_bounds=world_bounds,
        width=1280,
        height=720,
        title="Functional Test"
    )
    print("   ✓ Visualizer created")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Run simulation
print("\n4. Running simulation...")
frame_count = 0
total_collisions = 0

def render():
    global frame_count, total_collisions
    
    if frame_count >= max_frames:
        print("\n   Simulation complete!")
        vis.close()
        return
    
    # Step physics
    stats = sim.step()
    total_collisions += stats['num_collisions']
    
    # Get data
    positions = cp.asnumpy(sim.bodies.positions)
    radii = cp.asnumpy(sim.bodies.radii)
    
    # Check for overlaps
    num_overlaps = 0
    for i in range(num_balls):
        for j in range(i+1, num_balls):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist < radii[i] + radii[j] - 1e-6:
                num_overlaps += 1
    
    # Info text
    info = f"Frame: {frame_count}/{max_frames}\n"
    info += f"Collisions: {stats['num_collisions']}\n"
    info += f"Overlaps: {num_overlaps}\n"
    info += f"Sim FPS: {1000.0/stats['total_time']:.0f}"
    
    # Render
    try:
        vis.render(positions, radii, colors, info)
    except Exception as e:
        print(f"\n   ✗ Render error at frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()
        vis.close()
        return
    
    # Progress
    if frame_count % 30 == 0:
        print(f"   Frame {frame_count}: Collisions={stats['num_collisions']}, Overlaps={num_overlaps}")
    
    frame_count += 1

try:
    vis.set_render_function(render)
    vis.run()
except KeyboardInterrupt:
    print("\n   Interrupted by user")
except Exception as e:
    print(f"\n   ✗ Runtime error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n5. Results:")
print(f"   Total frames: {frame_count}")
print(f"   Total collisions: {total_collisions}")
print(f"   Average collisions/frame: {total_collisions/frame_count if frame_count > 0 else 0:.1f}")

if total_collisions > 0:
    print("\n✓ TEST PASSED: Collisions detected and visualized")
else:
    print("\n✗ TEST FAILED: No collisions detected")
    sys.exit(1)

print("\n" + "="*60)
print("Functional test complete!")
print("="*60)
