#!/usr/bin/env python3
"""
GPU Collision Detection - High-Quality Gravity Fall with OpenGL

Features:
- OpenGL realistic rendering
- 500 balls with varied properties
- High-quality video export
- Real-time collision detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
import colorsys
from src.simulator import PhysicsSimulator
from src.opengl_visualizer import OpenGLVisualizer, OpenGLVideoRecorder
from src.init_helper import generate_non_overlapping_positions, verify_no_overlaps
import time

def main():
    print("=" * 70)
    print("GPU Collision Detection - Gravity Fall (OpenGL)")
    print("=" * 70)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    NUM_OBJECTS = 150           # Number of balls
    # WORLD_BOUNDS = ((-20, 0, -20), (20, 40, 20))  # World boundaries
    WORLD_BOUNDS = ((-4, 0, -4), (4, 8, 4))  # World boundaries
    CELL_SIZE = 1.0             # Grid cell size (reduced for better collision detection)
    NUM_FRAMES = 480            # Total frames (@ 60fps)
    RECORD_VIDEO = False         # Record video
    
    print(f"\nConfiguration:")
    print(f"  Objects: {NUM_OBJECTS}")
    print(f"  World Bounds: {WORLD_BOUNDS}")
    print(f"  Cell Size: {CELL_SIZE}")
    print(f"  Total Frames: {NUM_FRAMES}")
    print(f"  Duration: {NUM_FRAMES/60:.1f} seconds @ 60fps")
    
    # ========================================================================
    # Initialize Simulator
    # ========================================================================
    print(f"\nInitializing simulator...")
    sim = PhysicsSimulator(
        num_objects=NUM_OBJECTS,
        world_bounds=WORLD_BOUNDS,
        cell_size=CELL_SIZE,
        device_id=0,
        dt=1.0 / 60.0,
        gravity=(0.0, -9.81, 0.0),
        damping=0.01
    )
    print("  ✓ Simulator created")
    
    # ========================================================================
    # Setup Scene - Generate Non-Overlapping Balls
    # ========================================================================
    print(f"\nSetting up scene...")
    
    # Generate varied radii (lognormal distribution for natural variety)
    np.random.seed(42)  # For reproducibility
    radii = np.random.lognormal(mean=-0.5, sigma=1.0, size=NUM_OBJECTS)
    radii = np.clip(radii, 0.15, 0.65).astype(np.float32)
    
    print(f"  Radius range: {radii.min():.2f} - {radii.max():.2f} m")
    
    # Generate non-overlapping positions
    print(f"  Generating {NUM_OBJECTS} non-overlapping spheres...")
    positions = generate_non_overlapping_positions(
        num_objects=NUM_OBJECTS,
        radii=radii,
        world_bounds=WORLD_BOUNDS,
        max_attempts=50
    )
    
    # Verify no overlaps
    num_overlaps, max_penetration = verify_no_overlaps(positions, radii)
    if num_overlaps > 0:
        print(f"  ⚠ Warning: {num_overlaps} initial overlaps detected")
        print(f"  Max penetration: {max_penetration:.6f} m")
    else:
        print(f"  ✓ All spheres initialized without overlaps")
    
    # Generate diverse initial velocities
    velocities = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
    for i in range(NUM_OBJECTS):
        roll = np.random.random()
        if roll < 0.5:
            # 50% with downward velocity
            velocities[i, 1] = np.random.uniform(-5, -1)
        elif roll < 0.8:
            # 30% with random velocity
            velocities[i] = np.random.uniform(-5, 5, 3)
        # 20% start at rest
    
    # Calculate masses (proportional to volume, with density variation)
    base_density = 1000.0  # kg/m³
    density_variation = np.random.uniform(0.8, 1.2, NUM_OBJECTS)
    volumes = (4/3) * np.pi * radii**3
    masses = (volumes * base_density * density_variation).astype(np.float32)
    
    print(f"  Mass range: {masses.min():.2f} - {masses.max():.2f} kg")
    
    # Restitution coefficients (bounciness)
    restitution = np.random.uniform(0.7, 0.99, NUM_OBJECTS).astype(np.float32)
    
    # Initialize simulator state
    sim.bodies.positions[:] = cp.asarray(positions)
    sim.bodies.velocities[:] = cp.asarray(velocities)
    sim.bodies.radii[:] = cp.asarray(radii)
    sim.bodies.masses[:] = cp.asarray(masses)
    sim.bodies.restitutions[:] = cp.asarray(restitution)
    
    print(f"  ✓ Initialized {NUM_OBJECTS} spheres")
    
    # ========================================================================
    # Generate Colors (HSV for rich variety)
    # ========================================================================
    colors = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
    for i in range(NUM_OBJECTS):
        # Use golden ratio for pleasant color distribution
        hue = (i * 0.618033988749895) % 1.0
        saturation = np.random.uniform(0.7, 1.0)
        value = np.random.uniform(0.8, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = rgb
    
    # ========================================================================
    # Initialize Visualizer
    # ========================================================================
    print(f"\nInitializing OpenGL visualizer...")
    vis = OpenGLVisualizer(
        world_bounds=WORLD_BOUNDS,
        width=1920,
        height=1080,
        title="GPU Collision Detection - Gravity Fall"
    )
    print("  ✓ Visualizer created")
    
    # ========================================================================
    # Initialize Video Recorder
    # ========================================================================
    recorder = None
    if RECORD_VIDEO:
        output_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'output', 
            'gravity_fall_opengl.mp4'
        )
        print(f"\nInitializing video recorder...")
        print(f"  Output: {output_path}")
        recorder = OpenGLVideoRecorder(output_path, 1920, 1080, fps=60)
        print("  ✓ Recorder ready")
    
    # ========================================================================
    # Simulation Loop
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("Starting simulation...")
    print("=" * 70)
    print("\nControls:")
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
    
    frame_count = 0
    total_collisions = 0
    start_time = time.time()
    
    # Progress reporting intervals
    report_interval = 30  # Report every 30 frames (0.5 seconds)
    
    def render():
        nonlocal frame_count, total_collisions
        
        if frame_count >= NUM_FRAMES:
            # Simulation complete
            elapsed = time.time() - start_time
            print(f"\n" + "=" * 70)
            print("Simulation completed!")
            print("=" * 70)
            print(f"\nStatistics:")
            print(f"  Total frames: {frame_count}")
            print(f"  Total time: {elapsed:.1f} seconds")
            print(f"  Average FPS: {frame_count/elapsed:.1f}")
            print(f"  Total collisions: {total_collisions}")
            print(f"  Average collisions/frame: {total_collisions/frame_count:.1f}")
            
            if recorder:
                print(f"\nFinalizing video...")
                recorder.release()
            
            vis.close()
            return
        
        # Don't step if paused
        if vis.paused:
            positions = cp.asnumpy(sim.bodies.positions)
            radii_cpu = cp.asnumpy(sim.bodies.radii)
            
            info = f"Frame: {frame_count}/{NUM_FRAMES}\n"
            info += "PAUSED (press Space)"
            
            vis.render(positions, radii_cpu, colors, info)
            return
        
        # Step physics
        stats = sim.step()
        total_collisions += stats['num_collisions']
        
        # Get current state
        positions = cp.asnumpy(sim.bodies.positions)
        radii_cpu = cp.asnumpy(sim.bodies.radii)
        
        # Check for overlaps (for debugging) - check ALL balls every frame
        num_overlaps = 0
        min_distance = float('inf')
        for i in range(NUM_OBJECTS):
            for j in range(i+1, NUM_OBJECTS):
                dist = np.linalg.norm(positions[j] - positions[i])
                min_distance = min(min_distance, dist)
                required_dist = radii_cpu[i] + radii_cpu[j]
                if dist < required_dist - 1e-6:
                    num_overlaps += 1
                    if frame_count % 30 == 0:  # Print overlaps periodically
                        print(f"  OVERLAP: Ball {i} <-> Ball {j}: dist={dist:.4f}m, required={required_dist:.4f}m, penetration={(required_dist-dist):.4f}m")
        
        # Prepare info text
        info = f"Frame: {frame_count}/{NUM_FRAMES}\n"
        info += f"Collisions: {stats['num_collisions']}\n"
        info += f"Sim FPS: {1000.0/stats['total_time']:.0f}\n"
        info += f"Min dist: {min_distance:.3f}m"
        if num_overlaps > 0:
            info += f"\n⚠ Overlaps: {num_overlaps}"
        
        # Render
        vis.render(positions, radii_cpu, colors, info)
        
        # Record frame
        if recorder:
            recorder.capture_frame()
        
        # Progress reporting
        if frame_count % report_interval == 0:
            elapsed = time.time() - start_time
            render_fps = frame_count / elapsed if elapsed > 0 else 0
            overlap_str = f", Overlaps={num_overlaps}" if num_overlaps > 0 else ""
            print(f"Frame {frame_count:4d}/{NUM_FRAMES}: "
                  f"Collisions={stats['num_collisions']}, "
                  f"MinDist={min_distance:.3f}m, "
                  f"SimFPS={1000.0/stats['total_time']:4.0f}, "
                  f"RenderFPS={render_fps:4.1f}{overlap_str}")
        
        frame_count += 1
    
    # Set render callback and run
    vis.set_render_function(render)
    
    try:
        vis.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if recorder:
            recorder.release()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        if recorder:
            recorder.release()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
