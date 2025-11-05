#!/usr/bin/env python3
"""Quick test with fewer balls to verify fixes"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
from src.visualizer import RealtimeVisualizer, VideoExporter

def main():
    print("=" * 60)
    print("Quick Test - 100 Balls")
    print("=" * 60)
    
    # Smaller test for quick verification
    num_objects = 100
    world_bounds = ((-10, 0, -10), (10, 20, 10))
    
    # Create simulator
    sim = PhysicsSimulator(
        num_objects=num_objects,
        world_bounds=world_bounds,
        cell_size=1.5,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)  # Gravity in -y direction
    )
    
    print(f"\nInitializing {num_objects} spheres...")
    
    # Generate varied radii (lognormal distribution)
    radii = np.random.lognormal(mean=-1.2, sigma=0.4, size=num_objects)
    radii = np.clip(radii, 0.15, 0.6).astype(np.float32)
    
    # Multi-layer initial positions
    positions = np.zeros((num_objects, 3), dtype=np.float32)
    for i in range(num_objects):
        layer = i % 4
        positions[i] = [
            np.random.uniform(-8, 8),
            12 + layer * 2,  # Heights: 12, 14, 16, 18
            np.random.uniform(-8, 8)
        ]
    
    # Varied initial velocities
    velocities = np.zeros((num_objects, 3), dtype=np.float32)
    for i in range(num_objects):
        if i % 3 == 0:  # 33% with downward velocity
            velocities[i, 1] = np.random.uniform(-2, -1)
        elif i % 3 == 1:  # 33% with random velocity
            velocities[i] = np.random.uniform(-1, 1, 3)
    
    # Masses proportional to volume
    masses = (4/3 * np.pi * radii**3 * 1000).astype(np.float32)
    
    # Restitution coefficients
    restitution = np.random.uniform(0.6, 0.9, num_objects).astype(np.float32)
    
    # Initialize
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    # Generate colors (HSV)
    import colorsys
    colors = np.zeros((num_objects, 3), dtype=np.float32)
    for i in range(num_objects):
        hue = np.random.uniform(0, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[i] = rgb
    
    print(f"  Radius range: {radii.min():.2f} - {radii.max():.2f}")
    print(f"  Mass range: {masses.min():.2f} - {masses.max():.2f} kg")
    
    # Create visualizer
    print("\nInitializing visualizer...")
    vis = RealtimeVisualizer(world_bounds)
    
    # Create video exporter
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'quick_test.mp4')
    video = VideoExporter(output_path, fps=60, resolution=(1280, 720))
    
    # Simulation parameters
    total_frames = 300  # 5 seconds
    
    print("\nStarting simulation...")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/60:.1f} seconds @ 60fps")
    print()
    
    for frame in range(total_frames):
        # Step simulation
        stats = sim.step()
        
        # Get data for visualization
        pos = cp.asnumpy(sim.bodies.positions)
        
        # Update visualization
        info = f"Frame {frame}/{total_frames}\n"
        info += f"Collisions: {stats['num_collisions']}\n"
        info += f"FPS: {1000.0 / stats['total_time']:.1f}"
        
        vis.update(pos, colors=colors, radii=radii, info_text=info)
        
        # Save frame
        video.add_frame_from_matplotlib(vis.fig)
        
        # Print progress
        if frame % 30 == 0 or frame == total_frames - 1:
            print(f"Frame {frame:3d}/{total_frames}: "
                  f"Collisions={stats['num_collisions']:3d}, "
                  f"FPS={1000.0/stats['total_time']:.0f}")
    
    print("\nSimulation completed!")
    
    # Close video
    video.release()
    vis.close()
    
    print(f"\nVideo saved to: {output_path}")
    print("\nVerify:")
    print("  1. Balls should fall downward (gravity in -y direction)")
    print("  2. In visualization, balls should fall toward bottom of screen")
    print("  3. No tunneling - balls should bounce off each other")
    print("  4. Collision count should be > 0 when balls interact")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
