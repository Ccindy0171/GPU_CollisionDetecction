#!/usr/bin/env python3
"""
Test OpenGL visualizer with falling balls simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
from src.opengl_visualizer import OpenGLVisualizer, OpenGLVideoRecorder
import colorsys

# Global variables
sim = None
vis = None
recorder = None
colors = None
frame_count = 0
max_frames = 600
recording = False


def init_simulation():
    """Initialize physics simulation"""
    global sim, vis, colors, recorder, recording
    
    # Simulation parameters
    num_balls = 50
    world_bounds = ((-15, 0, -15), (15, 30, 15))
    
    print("=" * 60)
    print("OpenGL Visualization Test")
    print("=" * 60)
    print(f"\nObjects: {num_balls}")
    print(f"World bounds: {world_bounds}")
    print(f"Total frames: {max_frames}")
    print(f"Duration: {max_frames/60:.1f} seconds @ 60fps\n")
    
    # Create simulator
    sim = PhysicsSimulator(
        num_objects=num_balls,
        world_bounds=world_bounds,
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)
    )
    
    # Generate random spheres
    radii = np.random.lognormal(mean=-1.2, sigma=0.4, size=num_balls)
    radii = np.clip(radii, 0.2, 0.8).astype(np.float32)
    
    # Multi-layer positions
    positions = np.zeros((num_balls, 3), dtype=np.float32)
    for i in range(num_balls):
        layer = i % 5
        positions[i] = [
            np.random.uniform(-12, 12),
            15 + layer * 3,
            np.random.uniform(-12, 12)
        ]
    
    # Check for initial overlaps and resolve
    print("Checking initial overlaps...")
    max_attempts = 100
    for attempt in range(max_attempts):
        overlaps = []
        for i in range(num_balls):
            for j in range(i+1, num_balls):
                dist = np.linalg.norm(positions[j] - positions[i])
                if dist < radii[i] + radii[j]:
                    overlaps.append((i, j, dist, radii[i] + radii[j]))
        
        if len(overlaps) == 0:
            break
        
        # Resolve overlaps
        for i, j, dist, min_dist in overlaps:
            if dist < 1e-6:
                # Balls at same position, move them apart
                positions[j] += np.random.randn(3) * min_dist
            else:
                # Push apart
                direction = (positions[j] - positions[i]) / dist
                overlap = min_dist - dist
                positions[i] -= direction * overlap * 0.5
                positions[j] += direction * overlap * 0.5
    
    if len(overlaps) > 0:
        print(f"Warning: Could not resolve all overlaps after {max_attempts} attempts")
    else:
        print(f"All overlaps resolved in {attempt+1} attempts")
    
    # Varied velocities
    velocities = np.zeros((num_balls, 3), dtype=np.float32)
    for i in range(num_balls):
        if i % 3 == 0:
            velocities[i, 1] = np.random.uniform(-2, -1)
    
    # Masses
    masses = (4/3 * np.pi * radii**3 * 1000).astype(np.float32)
    
    # Restitution
    restitution = np.random.uniform(0.6, 0.9, num_balls).astype(np.float32)
    
    # Initialize
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    # Generate colors (HSV for variety)
    colors = np.zeros((num_balls, 3), dtype=np.float32)
    for i in range(num_balls):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors[i] = rgb
    
    # Create visualizer
    vis = OpenGLVisualizer(
        world_bounds=world_bounds,
        width=1920,
        height=1080,
        title="GPU Collision Detection - OpenGL"
    )
    
    # Ask user if they want to record
    response = input("\nRecord video? (y/n): ").strip().lower()
    if response == 'y':
        recording = True
        output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'opengl_test.mp4')
        recorder = OpenGLVideoRecorder(output_path, 1920, 1080, fps=60)
        print(f"Recording to: {output_path}")
    
    print("\nStarting simulation...")
    print("Controls:")
    print("  Left Mouse: Rotate camera")
    print("  Right Mouse: Zoom")
    print("  Middle Mouse: Pan")
    print("  Space: Pause/Resume")
    print("  G: Toggle grid")
    print("  A: Toggle axes")
    print("  W: Toggle wireframe")
    print("  R: Reset camera")
    print("  Q/ESC: Quit")
    print()


def render():
    """Render callback"""
    global frame_count, max_frames
    
    if vis.paused:
        # Just re-render without stepping
        positions = cp.asnumpy(sim.bodies.positions)
        radii = cp.asnumpy(sim.bodies.radii)
        
        info = f"Frame: {frame_count}/{max_frames}\n"
        info += "PAUSED (press Space to resume)"
        
        vis.render(positions, radii, colors, info)
        return
    
    # Step simulation
    if frame_count < max_frames:
        stats = sim.step()
        
        # Get positions and radii
        positions = cp.asnumpy(sim.bodies.positions)
        radii = cp.asnumpy(sim.bodies.radii)
        
        # Prepare info text
        info = f"Frame: {frame_count}/{max_frames}\n"
        info += f"Collisions: {stats['num_collisions']}\n"
        info += f"Sim FPS: {1000.0/stats['total_time']:.0f}"
        
        # Render
        vis.render(positions, radii, colors, info)
        
        # Record frame if recording
        if recording and recorder is not None:
            recorder.capture_frame()
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Frame {frame_count:3d}/{max_frames}: "
                  f"Collisions={stats['num_collisions']:3d}, "
                  f"SimFPS={1000.0/stats['total_time']:.0f}")
        
        frame_count += 1
    
    else:
        # Simulation complete
        print("\nSimulation completed!")
        if recording and recorder is not None:
            recorder.release()
        vis.close()


def main():
    """Main function"""
    # Initialize
    init_simulation()
    
    # Set render callback
    vis.set_render_function(render)
    
    # Run
    vis.run()


if __name__ == "__main__":
    main()
