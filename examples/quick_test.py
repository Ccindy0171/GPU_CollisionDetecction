#!/usr/bin/env python3
"""
快速测试 - 小规模演示

运行一个快速的小规模仿真以验证系统功能。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src import (
    PhysicsSimulator,
    RealtimeVisualizer,
    VideoExporter,
    PerformanceMonitor
)


def main():
    """主函数"""
    print("=" * 70)
    print("Quick Test - Small Scale Demo")
    print("=" * 70)
    
    # 小规模配置
    NUM_OBJECTS = 1000
    WORLD_BOUNDS = ((-10, 0, -10), (10, 20, 10))
    CELL_SIZE = 2.0
    NUM_FRAMES = 120  # 2秒 @ 60fps
    
    print(f"\nConfiguration:")
    print(f"  Objects: {NUM_OBJECTS}")
    print(f"  World Bounds: {WORLD_BOUNDS}")
    print(f"  Frames: {NUM_FRAMES}")
    
    # 初始化仿真器
    print("\nInitializing simulator...")
    sim = PhysicsSimulator(
        world_bounds=WORLD_BOUNDS,
        cell_size=CELL_SIZE,
        num_objects=NUM_OBJECTS,
        gravity=(0, -9.8, 0)
    )
    
    # 设置场景
    print("Setting up scene...")
    positions = np.random.uniform(
        low=[WORLD_BOUNDS[0][0], 10, WORLD_BOUNDS[0][2]],
        high=[WORLD_BOUNDS[1][0], 18, WORLD_BOUNDS[1][2]],
        size=(NUM_OBJECTS, 3)
    ).astype(np.float32)
    sim.bodies.set_positions(positions)
    
    # 更多样化的初速度
    velocities = np.random.uniform(-2, 2, (NUM_OBJECTS, 3)).astype(np.float32)
    velocities[:, 1] = np.random.uniform(-3, 1, NUM_OBJECTS)  # Y方向有向下趋势
    sim.bodies.set_velocities(velocities)
    
    # 更大范围的半径
    radii_log = np.random.lognormal(mean=-1.2, sigma=0.4, size=NUM_OBJECTS)
    radii = np.clip(radii_log, 0.15, 0.6).astype(np.float32)
    sim.bodies.set_radii(radii)
    
    # 质量随半径变化
    density_variation = 800.0 + 600.0 * (1.0 - radii / radii.max())
    masses = (4.0/3.0 * np.pi * radii**3 * density_variation).astype(np.float32)
    sim.bodies.set_masses(masses)
    
    restitutions = np.random.uniform(0.4, 0.95, NUM_OBJECTS).astype(np.float32)
    sim.bodies.set_restitutions(restitutions)
    
    # 丰富多彩的颜色
    colors = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
    import colorsys
    hues = np.random.uniform(0, 1, NUM_OBJECTS)
    saturations = np.random.uniform(0.6, 1.0, NUM_OBJECTS)
    values = np.random.uniform(0.7, 1.0, NUM_OBJECTS)
    for i in range(NUM_OBJECTS):
        rgb = colorsys.hsv_to_rgb(hues[i], saturations[i], values[i])
        colors[i] = rgb
    sim.bodies.set_colors(colors)
    
    print(f"  Initialized {NUM_OBJECTS} spheres")
    
    # 初始化可视化
    print("\nInitializing visualizer...")
    visualizer = RealtimeVisualizer(
        world_bounds=WORLD_BOUNDS,
        figsize=(10, 8),
        title="Quick Test"
    )
    
    # 初始化视频导出
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'quick_test.mp4')
    video = VideoExporter(video_path, fps=60, resolution=(1280, 720))
    
    monitor = PerformanceMonitor()
    
    # 仿真循环
    print("\n" + "=" * 70)
    print("Starting simulation...")
    print("=" * 70)
    
    try:
        for frame in range(NUM_FRAMES):
            # 执行仿真步
            step_info = sim.step()
            
            # 记录性能
            monitor.record_metric('total_time', step_info['total_time'])
            monitor.record_metric('num_collisions', step_info['num_collisions'])
            
            # 打印进度
            if frame % 30 == 0 or frame == NUM_FRAMES - 1:
                stats = sim.get_stats()
                print(f"\nFrame {frame:3d}/{NUM_FRAMES}")
                print(f"  FPS: {stats['fps']:.1f}")
                print(f"  Collisions: {step_info['num_collisions']}")
                print(f"  Total Time: {step_info['total_time']:.2f} ms")
            
            # 更新可视化（每2帧）
            if frame % 2 == 0:
                data = sim.bodies.to_cpu()
                
                info_text = (
                    f"Frame: {frame}/{NUM_FRAMES}\n"
                    f"FPS: {stats['fps']:.1f}\n"
                    f"Collisions: {step_info['num_collisions']}"
                )
                
                visualizer.update(
                    positions=data['positions'],
                    colors=data['colors'],
                    radii=data['radii'],
                    info_text=info_text
                )
                
                video.add_frame_from_matplotlib(visualizer.fig)
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        print("\n" + "=" * 70)
        print("Simulation completed!")
        print("=" * 70)
        
        visualizer.close()
        video.release()
        
        print(f"\nVideo saved to: {video_path}")
        print("\nPerformance Statistics:")
        monitor.print_statistics()
        
        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)


if __name__ == '__main__':
    main()
