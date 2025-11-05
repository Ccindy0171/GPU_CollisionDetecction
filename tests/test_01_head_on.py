#!/usr/bin/env python3
"""
Simplest head-on collision test
Two balls moving towards each other with no gravity
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
    print("HEAD-ON COLLISION TEST - Two Balls")
    print("=" * 70)
    
    # Create simulator with 2 balls, NO GRAVITY
    num_objects = 2
    sim = PhysicsSimulator(
        num_objects=num_objects,
        world_bounds=((-5, -5, -5), (5, 5, 5)),
        cell_size=2.0,
        dt=1.0/60.0,
        gravity=(0, 0, 0)  # NO GRAVITY - isolate collision mechanics
    )
    
    print("\nTest Setup:")
    print("  2 balls moving towards each other")
    print("  Ball 1: Left side, moving RIGHT at 5 m/s")
    print("  Ball 2: Right side, moving LEFT at 5 m/s")
    print("  No gravity - pure collision test")
    print("  Expected: Balls collide and bounce back")
    
    # Initialize balls
    positions = np.array([
        [-2.0, 0.0, 0.0],  # Ball 1 on left
        [2.0, 0.0, 0.0],   # Ball 2 on right
    ], dtype=np.float32)
    
    velocities = np.array([
        [5.0, 0.0, 0.0],   # Ball 1 moving right
        [-5.0, 0.0, 0.0],  # Ball 2 moving left
    ], dtype=np.float32)
    
    radii = np.array([0.3, 0.3], dtype=np.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)
    restitutions = np.array([0.9, 0.9], dtype=np.float32)
    
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitutions)
    
    # Colors: red and blue
    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
    ], dtype=np.float32)
    
    print(f"\nInitial state:")
    print(f"  Ball 1: pos={positions[0]}, vel={velocities[0]}")
    print(f"  Ball 2: pos={positions[1]}, vel={velocities[1]}")
    print(f"  Distance: {np.linalg.norm(positions[1] - positions[0]):.3f}m")
    print(f"  Time to collision: ~{(np.linalg.norm(positions[1] - positions[0]) - 0.6) / 10:.2f}s")
    
    # Create visualizer
    world_bounds = ((-5, -5, -5), (5, 5, 5))
    vis = RealtimeVisualizer(world_bounds)
    
    # Create video
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'head_on_collision.mp4')
    video = VideoExporter(output_path, fps=60, resolution=(1280, 720))
    
    # Run simulation
    total_frames = 180  # 3 seconds
    collision_detected = False
    collision_frame = -1
    max_penetration = 0.0
    
    print("\nRunning simulation...")
    
    for frame in range(total_frames):
        # Step physics
        stats = sim.step()
        
        # Get current state
        pos = cp.asnumpy(sim.bodies.positions)
        vel = cp.asnumpy(sim.bodies.velocities)
        
        # Calculate distance
        dist = np.linalg.norm(pos[1] - pos[0])
        min_dist = radii[0] + radii[1]
        penetration = max(0, min_dist - dist)
        
        if penetration > max_penetration:
            max_penetration = penetration
        
        # Check for collision
        if stats['num_collisions'] > 0 and not collision_detected:
            collision_detected = True
            collision_frame = frame
            print(f"\n✓ Collision detected at frame {frame} ({frame/60:.2f}s)")
            print(f"  Distance: {dist:.4f}m")
            print(f"  Penetration: {penetration:.4f}m")
            print(f"  Ball 1: pos={pos[0]}, vel={vel[0]}")
            print(f"  Ball 2: pos={pos[1]}, vel={vel[1]}")
        
        # Update visualization
        info = f"Frame: {frame}/{total_frames}\n"
        info += f"Time: {frame/60:.2f}s\n"
        info += f"Distance: {dist:.3f}m\n"
        info += f"Min Dist: {min_dist:.3f}m\n"
        if penetration > 0:
            info += f"Penetration: {penetration:.4f}m\n"
        info += f"Collisions: {stats['num_collisions']}\n"
        info += f"Ball 1 vel: [{vel[0, 0]:+.2f}, {vel[0, 1]:+.2f}, {vel[0, 2]:+.2f}]\n"
        info += f"Ball 2 vel: [{vel[1, 0]:+.2f}, {vel[1, 1]:+.2f}, {vel[1, 2]:+.2f}]"
        
        vis.update(pos, colors=colors, radii=radii, info_text=info)
        video.add_frame_from_matplotlib(vis.fig)
        
        # Print key frames
        if frame % 30 == 0 or stats['num_collisions'] > 0:
            print(f"Frame {frame:3d}: dist={dist:.3f}m, pen={penetration:.4f}m, "
                  f"coll={stats['num_collisions']}, "
                  f"v1=[{vel[0,0]:+.2f},{vel[0,1]:+.2f},{vel[0,2]:+.2f}], "
                  f"v2=[{vel[1,0]:+.2f},{vel[1,1]:+.2f},{vel[1,2]:+.2f}]")
    
    # Final state
    pos_final = cp.asnumpy(sim.bodies.positions)
    vel_final = cp.asnumpy(sim.bodies.velocities)
    dist_final = np.linalg.norm(pos_final[1] - pos_final[0])
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Collision detected: {collision_detected}")
    if collision_detected:
        print(f"  At frame: {collision_frame} ({collision_frame/60:.2f}s)")
    print(f"Max penetration: {max_penetration:.4f}m")
    print(f"Final distance: {dist_final:.3f}m")
    print(f"Final Ball 1: pos={pos_final[0]}, vel={vel_final[0]}")
    print(f"Final Ball 2: pos={pos_final[1]}, vel={vel_final[1]}")
    
    # Check if velocities reversed
    vel_initial = velocities
    v1_reversed = vel_final[0, 0] < 0  # Should be negative now
    v2_reversed = vel_final[1, 0] > 0  # Should be positive now
    
    print(f"\nVelocity reversal check:")
    print(f"  Ball 1: {vel_initial[0, 0]:.2f} -> {vel_final[0, 0]:.2f} {'✓' if v1_reversed else '✗'}")
    print(f"  Ball 2: {vel_initial[1, 0]:.2f} -> {vel_final[1, 0]:.2f} {'✓' if v2_reversed else '✗'}")
    
    # Check momentum conservation
    momentum_initial = masses[0] * vel_initial[0] + masses[1] * vel_initial[1]
    momentum_final = masses[0] * vel_final[0] + masses[1] * vel_final[1]
    print(f"\nMomentum conservation:")
    print(f"  Initial: {momentum_initial}")
    print(f"  Final: {momentum_final}")
    print(f"  Conserved: {np.allclose(momentum_initial, momentum_final, atol=0.1)}")
    
    # Overall verdict
    print("\n" + "=" * 70)
    if collision_detected and v1_reversed and v2_reversed and max_penetration < 0.1:
        print("✓✓✓ TEST PASSED ✓✓✓")
        print("Collision mechanics are working correctly!")
    else:
        print("✗✗✗ TEST FAILED ✗✗✗")
        if not collision_detected:
            print("  - No collision detected!")
        if not (v1_reversed and v2_reversed):
            print("  - Velocities did not reverse properly!")
        if max_penetration >= 0.1:
            print(f"  - Excessive penetration: {max_penetration:.4f}m")
    print("=" * 70)
    
    # Close
    video.release()
    vis.close()
    
    print(f"\nVideo saved to: {output_path}")

if __name__ == "__main__":
    main()
