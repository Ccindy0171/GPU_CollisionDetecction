#!/usr/bin/env python3
"""
Single Ball Gravity Test - Verify basic physics works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src import PhysicsSimulator, RealtimeVisualizer, VideoExporter

def main():
    print("=" * 70)
    print("Single Ball Gravity Test")
    print("=" * 70)
    
    # Simple config - 1 ball
    NUM_OBJECTS = 1
    WORLD_BOUNDS = ((-5, 0, -5), (5, 10, 5))
    
    print(f"\nTest Setup:")
    print(f"  Number of balls: {NUM_OBJECTS}")
    print(f"  Initial position: (0, 8, 0)")
    print(f"  Initial velocity: (0, 0, 0)")
    print(f"  Gravity: (0, -9.8, 0)")
    print(f"  Ground at Y=0")
    
    # Initialize simulator
    sim = PhysicsSimulator(
        num_objects=NUM_OBJECTS,
        world_bounds=WORLD_BOUNDS,
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0.0, -9.8, 0.0),
        damping=0.0
    )
    
    # Set up a single ball high up
    positions = np.array([[0.0, 8.0, 0.0]], dtype=np.float32)
    sim.bodies.set_positions(positions)
    
    velocities = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    sim.bodies.set_velocities(velocities)
    
    radii = np.array([0.3], dtype=np.float32)
    sim.bodies.set_radii(radii)
    
    masses = np.array([1.0], dtype=np.float32)
    sim.bodies.set_masses(masses)
    
    restitutions = np.array([0.8], dtype=np.float32)
    sim.bodies.set_restitutions(restitutions)
    
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Red ball
    sim.bodies.set_colors(colors)
    
    print("\n" + "=" * 70)
    print("Running Simulation (120 frames = 2 seconds)")
    print("=" * 70)
    
    # Initialize visualizer
    visualizer = RealtimeVisualizer(
        world_bounds=WORLD_BOUNDS,
        figsize=(10, 8),
        title="Single Ball Test"
    )
    
    # Initialize video
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'single_ball_test.mp4')
    video = VideoExporter(video_path, fps=60, resolution=(1280, 720))
    
    # Track data for analysis
    positions_history = []
    velocities_history = []
    
    # Run simulation
    for frame in range(120):
        step_info = sim.step()
        
        data = sim.bodies.to_cpu()
        pos = data['positions'][0]
        vel = data['velocities'][0]
        
        positions_history.append(pos.copy())
        velocities_history.append(vel.copy())
        
        # Print every 10 frames
        if frame % 10 == 0:
            print(f"\nFrame {frame:3d}:")
            print(f"  Position: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            print(f"  Velocity: [{vel[0]:6.2f}, {vel[1]:6.2f}, {vel[2]:6.2f}]")
            print(f"  Speed: {np.linalg.norm(vel):.2f} m/s")
        
        # Update visualization every 2 frames
        if frame % 2 == 0:
            info_text = (
                f"Frame: {frame}/120\n"
                f"Pos Y: {pos[1]:.2f}\n"
                f"Vel Y: {vel[1]:.2f}\n"
                f"Collisions: {step_info['num_collisions']}"
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
    print("Analysis")
    print("=" * 70)
    
    positions_history = np.array(positions_history)
    velocities_history = np.array(velocities_history)
    
    # Find when ball hits ground (Y velocity changes sign significantly)
    vel_y = velocities_history[:, 1]
    bounces = []
    for i in range(1, len(vel_y)):
        if vel_y[i-1] < -1.0 and vel_y[i] > 1.0:  # Sign change with significant velocity
            bounces.append(i)
    
    print(f"\nBall behavior:")
    print(f"  Initial Y position: {positions_history[0, 1]:.2f}")
    print(f"  Minimum Y position: {positions_history[:, 1].min():.2f}")
    print(f"  Maximum downward velocity: {vel_y.min():.2f} m/s")
    print(f"  Number of bounces detected: {len(bounces)}")
    
    if len(bounces) > 0:
        print(f"  First bounce at frame: {bounces[0]}")
        for i, bounce_frame in enumerate(bounces[:3]):
            print(f"    Bounce {i+1}: frame {bounce_frame}, pos_y={positions_history[bounce_frame, 1]:.2f}")
    
    # Check if physics is working
    fell = positions_history[0, 1] - positions_history[:, 1].min() > 1.0
    accelerated = vel_y.min() < -2.0
    bounced = len(bounces) > 0
    
    print(f"\nPhysics Verification:")
    print(f"  Ball fell significantly: {'YES' if fell else 'NO'}")
    print(f"  Ball accelerated downward: {'YES' if accelerated else 'NO'}")
    print(f"  Ball bounced: {'YES' if bounced else 'NO'}")
    
    if fell and accelerated and bounced:
        print(f"\n✓ SUCCESS: Physics is working correctly!")
    else:
        print(f"\n✗ FAILURE: Physics is NOT working correctly!")
        print(f"\nDebugging info:")
        print(f"  Gravity vector: {sim.gravity.get()}")
        print(f"  dt: {sim.dt}")
        print(f"  damping: {sim.damping}")
    
    print(f"\nVideo saved to: {video_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
