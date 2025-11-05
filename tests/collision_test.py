#!/usr/bin/env python3
"""
Collision Test - Verify collision detection works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src import PhysicsSimulator, RealtimeVisualizer, VideoExporter

def main():
    print("=" * 70)
    print("Collision Detection Test")
    print("=" * 70)
    
    # Two balls placed very close together
    NUM_OBJECTS = 2
    WORLD_BOUNDS = ((-5, 0, -5), (5, 10, 5))
    
    print(f"\nTest Setup:")
    print(f"  Two balls placed close together")
    print(f"  Ball 1 at (0, 5, 0)")
    print(f"  Ball 2 at (0.5, 5, 0) - should collide!")
    
    # Initialize simulator
    sim = PhysicsSimulator(
        num_objects=NUM_OBJECTS,
        world_bounds=WORLD_BOUNDS,
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0.0, -9.8, 0.0),
        damping=0.0
    )
    
    # Place two balls close together (overlapping)
    positions = np.array([
        [0.0, 5.0, 0.0],   # Ball 1
        [0.5, 5.0, 0.0]    # Ball 2 - only 0.5m apart, radii = 0.3 each
    ], dtype=np.float32)
    sim.bodies.set_positions(positions)
    
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    sim.bodies.set_velocities(velocities)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    sim.bodies.set_radii(radii)
    
    masses = np.array([1.0, 1.0], dtype=np.float32)
    sim.bodies.set_masses(masses)
    
    restitutions = np.array([0.8, 0.8], dtype=np.float32)
    sim.bodies.set_restitutions(restitutions)
    
    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0]   # Blue
    ], dtype=np.float32)
    sim.bodies.set_colors(colors)
    
    print("\n" + "=" * 70)
    print("Running Simulation (60 frames = 1 second)")
    print("=" * 70)
    
    # Initialize visualizer
    visualizer = RealtimeVisualizer(
        world_bounds=WORLD_BOUNDS,
        figsize=(10, 8),
        title="Collision Test - 2 Balls"
    )
    
    # Initialize video
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'collision_test.mp4')
    video = VideoExporter(video_path, fps=60, resolution=(1280, 720))
    
    total_collisions = 0
    collision_frames = []
    
    # Run simulation
    for frame in range(60):
        step_info = sim.step()
        
        if step_info['num_collisions'] > 0:
            total_collisions += step_info['num_collisions']
            collision_frames.append(frame)
        
        data = sim.bodies.to_cpu()
        
        # Print every 10 frames
        if frame % 10 == 0 or step_info['num_collisions'] > 0:
            pos1 = data['positions'][0]
            pos2 = data['positions'][1]
            vel1 = data['velocities'][0]
            vel2 = data['velocities'][1]
            distance = np.linalg.norm(pos1 - pos2)
            
            print(f"\nFrame {frame:3d}:")
            print(f"  Ball 1: pos=[{pos1[0]:5.2f}, {pos1[1]:5.2f}, {pos1[2]:5.2f}], "
                  f"vel=[{vel1[0]:5.2f}, {vel1[1]:5.2f}, {vel1[2]:5.2f}]")
            print(f"  Ball 2: pos=[{pos2[0]:5.2f}, {pos2[1]:5.2f}, {pos2[2]:5.2f}], "
                  f"vel=[{vel2[0]:5.2f}, {vel2[1]:5.2f}, {vel2[2]:5.2f}]")
            print(f"  Distance: {distance:.3f}m (should be > {radii[0] + radii[1]:.1f}m when separated)")
            print(f"  Collisions: {step_info['num_collisions']}")
        
        # Update visualization every 2 frames
        if frame % 2 == 0:
            info_text = (
                f"Frame: {frame}/60\n"
                f"Distance: {np.linalg.norm(data['positions'][0] - data['positions'][1]):.2f}m\n"
                f"Collisions: {step_info['num_collisions']}\n"
                f"Total: {total_collisions}"
            )
            
            visualizer.update(
                positions=data['positions'],
                colors=data['colors'],
                radii=data['radii'],
                info_text=info_text
            )
            
            video.add_frame_from_matplotlib(visualizer.fig)
    
    visualizer.close()
    video.release()
    
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    
    print(f"\nCollision Statistics:")
    print(f"  Total collisions detected: {total_collisions}")
    print(f"  Frames with collisions: {len(collision_frames)}")
    if collision_frames:
        print(f"  Collision frames: {collision_frames[:10]}{'...' if len(collision_frames) > 10 else ''}")
    
    data = sim.bodies.to_cpu()
    final_distance = np.linalg.norm(data['positions'][0] - data['positions'][1])
    min_distance = radii[0] + radii[1]
    
    print(f"\nFinal State:")
    print(f"  Distance between balls: {final_distance:.3f}m")
    print(f"  Minimum distance (sum of radii): {min_distance:.3f}m")
    print(f"  Separated: {final_distance >= min_distance}")
    
    if total_collisions > 0:
        print(f"\n✓ SUCCESS: Collision detection is working!")
    else:
        print(f"\n✗ FAILURE: No collisions detected!")
        print(f"\nDebugging info:")
        print(f"  Initial distance: {np.linalg.norm(positions[0] - positions[1]):.3f}m")
        print(f"  Sum of radii: {min_distance:.3f}m")
        print(f"  Should overlap: {np.linalg.norm(positions[0] - positions[1]) < min_distance}")
    
    print(f"\nVideo saved to: {video_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
