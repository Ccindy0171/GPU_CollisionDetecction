#!/usr/bin/env python3
"""
Test 04: Large Scale Test - 100 Balls
Stress test with many balls to verify no tunneling at scale
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
from src.visualizer import RealtimeVisualizer, VideoExporter
import colorsys

def main():
    print("=" * 70)
    print("LARGE SCALE TEST - 100 Balls")
    print("=" * 70)
    
    # Create simulator
    num_objects = 100
    sim = PhysicsSimulator(
        num_objects=num_objects,
        world_bounds=((-10, 0, -10), (10, 20, 10)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)
    )
    
    print("\nTest Setup:")
    print(f"  {num_objects} balls arranged in layers")
    print("  Varied sizes and masses")
    print("  Gravity: -9.81 m/s²")
    print("  Expected: Balls form pile without tunneling")
    
    # Generate non-overlapping initial positions
    print("\nGenerating non-overlapping initial positions...")
    
    # Use lognormal distribution for radii
    radii = np.random.lognormal(mean=-1.2, sigma=0.4, size=num_objects)
    radii = np.clip(radii, 0.2, 0.5).astype(np.float32)
    
    # Place balls in layers, with spacing to avoid initial overlap
    positions = []
    grid_size = int(np.ceil(np.sqrt(num_objects)))
    spacing = 1.2  # Space between ball centers
    
    for i in range(num_objects):
        layer = i // (grid_size * grid_size)
        in_layer = i % (grid_size * grid_size)
        row = in_layer // grid_size
        col = in_layer % grid_size
        
        x = (col - grid_size/2) * spacing
        z = (row - grid_size/2) * spacing
        y = 12 + layer * spacing
        
        positions.append([x, y, z])
    
    positions = np.array(positions, dtype=np.float32)
    
    # Check for initial overlaps
    initial_overlaps = 0
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist < radii[i] + radii[j]:
                initial_overlaps += 1
    
    print(f"Initial overlapping pairs: {initial_overlaps}")
    if initial_overlaps > 0:
        print("WARNING: Some balls are initially overlapping!")
    
    # Initialize simulation
    velocities = np.zeros((num_objects, 3), dtype=np.float32)
    masses = (4/3 * np.pi * radii**3 * 1000).astype(np.float32)
    restitutions = np.random.uniform(0.6, 0.8, num_objects).astype(np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    # Colors
    colors = np.zeros((num_objects, 3), dtype=np.float32)
    for i in range(num_objects):
        h = (i / num_objects + 0.1) % 1.0
        rgb = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        colors[i] = rgb
    
    # Create visualizer
    world_bounds = ((-10, 0, -10), (10, 20, 10))
    vis = RealtimeVisualizer(world_bounds)
    
    # Create video
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'large_scale_test.mp4')
    video = VideoExporter(output_path, fps=60, resolution=(1280, 720))
    
    # Run simulation
    total_frames = 600  # 10 seconds
    
    print("\nRunning simulation...")
    
    max_penetration_ever = 0.0
    collision_history = []
    overlap_history = []
    
    for frame in range(total_frames):
        # Step physics
        stats = sim.step()
        
        # Get current state
        pos = cp.asnumpy(sim.bodies.positions)
        
        # Calculate overlaps (sample every 10 frames to save time)
        max_penetration = 0.0
        num_overlaps = 0
        
        if frame % 10 == 0:
            for i in range(num_objects):
                for j in range(i+1, num_objects):
                    dist = np.linalg.norm(pos[j] - pos[i])
                    min_dist = radii[i] + radii[j]
                    overlap = max(0, min_dist - dist)
                    if overlap > 0:
                        max_penetration = max(max_penetration, overlap)
                        num_overlaps += 1
            
            if max_penetration > max_penetration_ever:
                max_penetration_ever = max_penetration
            
            overlap_history.append((frame, num_overlaps, max_penetration))
        
        collision_history.append(stats['num_collisions'])
        
        # Update visualization (every 2 frames to speed up)
        if frame % 2 == 0:
            min_y = np.min(pos[:, 1])
            max_y = np.max(pos[:, 1])
            
            info = f"Frame: {frame}/{total_frames} ({frame/60:.1f}s)\n"
            info += f"Collisions: {stats['num_collisions']}\n"
            if frame % 10 == 0:
                info += f"Overlaps: {num_overlaps}\n"
                if max_penetration > 0:
                    info += f"Max pen: {max_penetration:.4f}m\n"
            info += f"Height: {min_y:.1f} - {max_y:.1f}m\n"
            info += f"FPS: {1000.0/stats['total_time']:.0f}"
            
            vis.update(pos, colors=colors, radii=radii, info_text=info)
            video.add_frame_from_matplotlib(vis.fig)
        
        # Print progress
        if frame % 60 == 0:
            print(f"Frame {frame:3d} ({frame/60:.1f}s): coll={stats['num_collisions']:3d}, "
                  f"FPS={1000.0/stats['total_time']:.0f}")
    
    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Collision statistics
    total_collisions = sum(collision_history)
    avg_collisions = np.mean(collision_history)
    max_collisions = np.max(collision_history)
    
    print(f"\nCollision statistics:")
    print(f"  Total collisions: {total_collisions}")
    print(f"  Average per frame: {avg_collisions:.1f}")
    print(f"  Maximum in one frame: {max_collisions}")
    
    # Penetration statistics
    print(f"\nPenetration statistics:")
    print(f"  Max penetration ever: {max_penetration_ever:.4f}m")
    
    # Show overlap timeline
    print(f"\nOverlap timeline (sampled):")
    for i in range(0, len(overlap_history), len(overlap_history)//10):
        frame, n, m = overlap_history[i]
        print(f"  Frame {frame:3d}: {n} overlaps, max={m:.4f}m")
    
    if len(overlap_history) > 0:
        final_overlaps = overlap_history[-1][1]
        final_max_pen = overlap_history[-1][2]
        print(f"\nFinal state:")
        print(f"  Overlapping pairs: {final_overlaps}")
        print(f"  Max penetration: {final_max_pen:.4f}m")
    
    # Final verdict
    print("\n" + "=" * 70)
    if max_penetration_ever < 0.05 and (len(overlap_history) == 0 or overlap_history[-1][1] == 0):
        print("✓✓✓ TEST PASSED ✓✓✓")
        print("No significant tunneling!")
        print("System handles 100 balls correctly!")
    elif max_penetration_ever < 0.15:
        print("✓ TEST MOSTLY PASSED")
        print(f"Max penetration {max_penetration_ever:.4f}m is acceptable")
    else:
        print("✗ TEST FAILED")
        print(f"Excessive penetration: {max_penetration_ever:.4f}m")
    print("=" * 70)
    
    # Close
    video.release()
    vis.close()
    
    print(f"\nVideo saved to: {output_path}")

if __name__ == "__main__":
    main()
