#!/usr/bin/env python3
"""Quick test to verify camera angle and no initial overlaps"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src.simulator import PhysicsSimulator
from src.visualizer import RealtimeVisualizer, VideoExporter
from src.init_helper import generate_non_overlapping_positions, verify_no_overlaps

def main():
    print("=" * 60)
    print("Camera Angle and Initial Overlap Test")
    print("=" * 60)
    
    # Small test with 50 balls
    num_objects = 50
    world_bounds = ((-10, 0, -10), (10, 20, 10))
    
    # Create simulator
    sim = PhysicsSimulator(
        num_objects=num_objects,
        world_bounds=world_bounds,
        cell_size=1.5,
        dt=1.0/60.0,
        gravity=(0, -9.81, 0)
    )
    
    print(f"\n1. GENERATING NON-OVERLAPPING POSITIONS")
    
    # Generate radii first
    radii = np.random.uniform(0.2, 0.5, num_objects).astype(np.float32)
    
    # Generate non-overlapping positions
    print("   Generating positions...")
    positions = generate_non_overlapping_positions(
        num_objects,
        radii,
        world_bounds,
        max_attempts=100
    )
    
    # Verify no overlaps
    num_overlaps, max_penetration = verify_no_overlaps(positions, radii)
    print(f"   Initial overlaps: {num_overlaps}")
    if num_overlaps > 0:
        print(f"   Max penetration: {max_penetration:.3f}m")
        print(f"   ✗ FAIL: Initial overlaps detected!")
        return False
    else:
        print(f"   ✓ SUCCESS: No initial overlaps!")
    
    # Set up simulator
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.velocities[:] = cp.zeros((num_objects, 3), dtype=cp.float32)
    sim.bodies.masses[:] = cp.asarray((radii * 1000).astype(np.float32))
    sim.bodies.restitutions[:] = cp.full(num_objects, 0.8, dtype=cp.float32)
    
    # Generate colors
    import colorsys
    colors = np.zeros((num_objects, 3), dtype=np.float32)
    for i in range(num_objects):
        hue = np.random.uniform(0, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[i] = rgb
    
    print(f"\n2. TESTING CAMERA ANGLE")
    print(f"   Gravity direction: -Y (downward)")
    print(f"   Expected: Balls should fall toward bottom of screen")
    print(f"   Camera: elev=15, azim=45 (side view)")
    
    # Create visualizer
    vis = RealtimeVisualizer(world_bounds)
    
    # Create video
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'camera_overlap_test.mp4')
    video = VideoExporter(output_path, fps=60, resolution=(1280, 720))
    
    print(f"\n3. RUNNING SIMULATION (120 frames = 2 seconds)")
    
    overlap_count_per_frame = []
    
    for frame in range(120):
        # Step simulation
        stats = sim.step()
        
        # Check for overlaps
        pos = cp.asnumpy(sim.bodies.positions)
        num_overlaps, max_pen = verify_no_overlaps(pos, radii)
        overlap_count_per_frame.append(num_overlaps)
        
        # Update visualization
        info = f"Frame {frame}/120\n"
        info += f"Collisions: {stats['num_collisions']}\n"
        info += f"Overlaps: {num_overlaps}\n"
        info += f"Max Penetration: {max_pen:.3f}m"
        
        vis.update(pos, colors=colors, radii=radii, info_text=info)
        video.add_frame_from_matplotlib(vis.fig)
        
        if frame % 30 == 0:
            print(f"   Frame {frame:3d}: Collisions={stats['num_collisions']:3d}, Overlaps={num_overlaps}")
    
    print(f"\n4. RESULTS")
    
    # Check if overlaps increased
    initial_overlaps = overlap_count_per_frame[0]
    final_overlaps = overlap_count_per_frame[-1]
    max_overlaps = max(overlap_count_per_frame)
    
    print(f"   Initial overlaps: {initial_overlaps}")
    print(f"   Final overlaps: {final_overlaps}")
    print(f"   Max overlaps during simulation: {max_overlaps}")
    
    if max_overlaps == 0:
        print(f"   ✓ SUCCESS: No overlaps throughout simulation!")
    elif final_overlaps <= initial_overlaps:
        print(f"   ✓ PASS: Overlaps did not increase")
    else:
        print(f"   ✗ FAIL: Overlaps increased during simulation!")
    
    # Close
    video.release()
    vis.close()
    
    print(f"\nVideo saved to: {output_path}")
    print(f"\nPlease verify:")
    print(f"  1. Balls fall toward BOTTOM of screen (not left/right)")
    print(f"  2. No persistent overlapping/clumping")
    print(f"  3. Clean collisions and bouncing")
    
    return max_overlaps <= initial_overlaps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
