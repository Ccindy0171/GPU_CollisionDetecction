#!/usr/bin/env python3
"""
示例1：重力下落场景

模拟大量小球在重力作用下从上方下落，与地面和彼此碰撞的场景。
这个示例展示了完整的碰撞检测、物理仿真和动画导出功能。

运行方式:
    python examples/gravity_fall.py

输出:
    - 实时3D可视化窗口
    - output/gravity_fall.mp4 视频文件
    - output/gravity_fall_performance.png 性能分析图
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
from src import (
    PhysicsSimulator,
    RealtimeVisualizer,
    VideoExporter,
    PerformanceMonitor
)
from src.init_helper import generate_non_overlapping_positions, verify_no_overlaps


def main():
    """主函数"""
    print("=" * 70)
    print("GPU Collision Detection - Gravity Fall Simulation")
    print("=" * 70)
    
    # ========================================================================
    # 配置参数（针对RTX 3050优化）
    # ========================================================================
    NUM_OBJECTS = 800          # 物体数量
    WORLD_BOUNDS = ((-25, 0, -25), (25, 50, 25))  # 世界边界
    CELL_SIZE = 2.5             # 网格单元大小
    NUM_FRAMES = 2000            # 仿真帧数（10秒 @ 60fps）
    SAVE_VIDEO = True           # 是否保存视频
    SHOW_REALTIME = True        # 是否显示实时可视化
    FRAME_SKIP = 2              # 可视化跳帧（每N帧显示一次）
    
    print(f"\nConfiguration:")
    print(f"  Objects: {NUM_OBJECTS}")
    print(f"  World Bounds: {WORLD_BOUNDS}")
    print(f"  Cell Size: {CELL_SIZE}")
    print(f"  Total Frames: {NUM_FRAMES}")
    print(f"  Duration: {NUM_FRAMES / 60:.1f} seconds @ 60fps")
    
    # ========================================================================
    # 初始化仿真器
    # ========================================================================
    print("\nInitializing simulator...")
    
    sim = PhysicsSimulator(
        num_objects=NUM_OBJECTS,
        world_bounds=WORLD_BOUNDS,
        cell_size=CELL_SIZE,
        dt=1.0/60.0,
        gravity=(0.0, -9.8, 0.0),
        damping=0.01
    )
    
    # ========================================================================
    # 场景设置：随机分布的小球从上方下落
    # ========================================================================
    print("Setting up scene...")
    
    with cp.cuda.Device(sim.device_id):
        # 先生成半径（因为需要用于生成不重叠的位置）
        # 半径：更大范围的大小差异
        # 使用对数正态分布获得更自然的大小分布
        radii_log = np.random.lognormal(mean=-1.0, sigma=0.5, size=NUM_OBJECTS)
        radii = np.clip(radii_log, 0.15, 0.8).astype(np.float32)  # 更大的范围
        
        # 位置：使用helper函数生成不重叠的初始位置
        print("  Generating non-overlapping initial positions...")
        positions = generate_non_overlapping_positions(
            NUM_OBJECTS, 
            radii, 
            WORLD_BOUNDS,
            max_attempts=100
        )
        
        # 验证没有重叠
        num_overlaps, max_penetration = verify_no_overlaps(positions, radii)
        print(f"  Initial overlaps: {num_overlaps}")
        if num_overlaps > 0:
            print(f"  Max penetration: {max_penetration:.3f}m")
        
        sim.bodies.set_positions(positions)
        
        # 设置半径（已经在上面生成）
        sim.bodies.set_radii(radii)
        
        # 速度：更多样化的初速度
        velocities = np.random.uniform(-3, 3, (NUM_OBJECTS, 3)).astype(np.float32)
        # 一部分小球向下，一部分静止，一部分随机方向
        velocity_types = np.random.rand(NUM_OBJECTS)
        velocities[velocity_types < 0.6, 1] = np.random.uniform(-5, -1, np.sum(velocity_types < 0.6))  # 60%向下
        velocities[(velocity_types >= 0.6) & (velocity_types < 0.8), :] = 0  # 20%静止
        # 剩余20%保持随机速度
        sim.bodies.set_velocities(velocities)
        
        # 质量：与体积成正比，但密度有所不同
        # 大球密度稍低，小球密度稍高（更真实的物理）
        density_base = 1000.0  # kg/m^3
        density_variation = 500.0 + 1000.0 * (1.0 - radii / radii.max())  # 小球更重
        masses = (4.0/3.0 * np.pi * radii**3 * density_variation).astype(np.float32)
        sim.bodies.set_masses(masses)
        
        # 弹性系数：更多样化
        restitutions = np.random.uniform(0.3, 0.95, NUM_OBJECTS).astype(np.float32)
        sim.bodies.set_restitutions(restitutions)
        
        # 颜色：丰富多彩的随机颜色
        colors = np.zeros((NUM_OBJECTS, 3), dtype=np.float32)
        # 使用HSV色彩空间生成更丰富的颜色
        hues = np.random.uniform(0, 1, NUM_OBJECTS)
        saturations = np.random.uniform(0.6, 1.0, NUM_OBJECTS)
        values = np.random.uniform(0.7, 1.0, NUM_OBJECTS)
        
        # 转换HSV到RGB
        import colorsys
        for i in range(NUM_OBJECTS):
            rgb = colorsys.hsv_to_rgb(hues[i], saturations[i], values[i])
            colors[i] = rgb
        
        sim.bodies.set_colors(colors)
    
    print(f"  Initialized {NUM_OBJECTS} spheres")
    print(f"  Radius range: {radii.min():.2f} - {radii.max():.2f}")
    print(f"  Mass range: {masses.min():.2f} - {masses.max():.2f} kg")
    
    # ========================================================================
    # 初始化可视化和性能监控
    # ========================================================================
    visualizer = None
    if SHOW_REALTIME or SAVE_VIDEO:
        print("\nInitializing visualizer...")
        visualizer = RealtimeVisualizer(
            world_bounds=WORLD_BOUNDS,
            figsize=(12, 10),
            title="Gravity Fall Simulation"
        )
    
    video = None
    if SAVE_VIDEO:
        print("Initializing video exporter...")
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, 'gravity_fall.mp4')
        video = VideoExporter(video_path, fps=60, resolution=(1920, 1080))
    
    monitor = PerformanceMonitor()
    
    # ========================================================================
    # 仿真循环
    # ========================================================================
    print("\n" + "=" * 70)
    print("Starting simulation...")
    print("=" * 70)
    
    try:
        for frame in range(NUM_FRAMES):
            # 执行仿真步
            step_info = sim.step()
            
            # 记录性能指标
            monitor.record_metric('total_time', step_info['total_time'])
            monitor.record_metric('grid_build_time', step_info['grid_build_time'])
            monitor.record_metric('collision_detect_time', step_info['collision_detect_time'])
            monitor.record_metric('collision_resolve_time', step_info['collision_resolve_time'])
            monitor.record_metric('integrate_time', step_info['integrate_time'])
            monitor.record_metric('num_collisions', step_info['num_collisions'])
            
            # 打印进度
            if frame % 60 == 0 or frame == NUM_FRAMES - 1:
                stats = sim.get_stats()
                sys_info = sim.get_system_info()
                
                print(f"\nFrame {frame:4d}/{NUM_FRAMES}")
                print(f"  FPS: {stats['fps']:.1f}")
                print(f"  Collisions: {step_info['num_collisions']}")
                print(f"  Grid Build: {stats['grid_build_time']:.2f} ms")
                print(f"  Collision Detect: {stats['collision_detect_time']:.2f} ms")
                print(f"  Collision Resolve: {stats['collision_resolve_time']:.2f} ms")
                print(f"  Integration: {stats['integrate_time']:.2f} ms")
                print(f"  Total: {stats['total_time']:.2f} ms")
                print(f"  Grid Occupancy: {sys_info['cell_occupancy']*100:.1f}%")
                print(f"  Kinetic Energy: {sys_info['kinetic_energy']:.2f} J")
            
            # 更新可视化
            if visualizer and frame % FRAME_SKIP == 0:
                data = sim.bodies.to_cpu()
                
                # 准备信息文本
                info_text = (
                    f"Frame: {frame}/{NUM_FRAMES}\n"
                    f"FPS: {stats['fps']:.1f}\n"
                    f"Collisions: {step_info['num_collisions']}\n"
                    f"Objects: {NUM_OBJECTS}"
                )
                
                visualizer.update(
                    positions=data['positions'],
                    colors=data['colors'],
                    radii=data['radii'],
                    info_text=info_text
                )
                
                # 保存到视频
                if video:
                    video.add_frame_from_matplotlib(visualizer.fig)
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        # ====================================================================
        # 清理和输出结果
        # ====================================================================
        print("\n" + "=" * 70)
        print("Simulation completed!")
        print("=" * 70)
        
        # 关闭可视化
        if visualizer:
            visualizer.close()
        
        # 完成视频导出
        if video:
            video.release()
            print(f"\nVideo saved to: {video_path}")
        
        # 输出性能统计
        print("\nPerformance Statistics:")
        monitor.print_statistics()
        
        # 保存性能图表
        if len(monitor.metrics) > 0:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
            perf_plot_path = os.path.join(output_dir, 'gravity_fall_performance.png')
            
            monitor.plot_timeline(
                metrics=['total_time', 'grid_build_time', 'collision_detect_time', 
                        'collision_resolve_time', 'integrate_time'],
                save_path=perf_plot_path
            )
            print(f"\nPerformance plot saved to: {perf_plot_path}")
            
            # 导出CSV数据
            csv_path = os.path.join(output_dir, 'gravity_fall_metrics.csv')
            monitor.export_to_csv(csv_path)
            print(f"Performance data saved to: {csv_path}")
        
        # 输出系统信息
        sys_info = sim.get_system_info()
        print("\nFinal System State:")
        print(f"  Total Objects: {sys_info['num_objects']}")
        print(f"  Grid Resolution: {sys_info['grid_resolution']}")
        print(f"  Total Cells: {sys_info['total_cells']}")
        print(f"  Occupied Cells: {sys_info['occupied_cells']} "
              f"({sys_info['cell_occupancy']*100:.1f}%)")
        print(f"  Kinetic Energy: {sys_info['kinetic_energy']:.2f} J")
        print(f"  Momentum: {sys_info['momentum']}")
        
        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)


if __name__ == '__main__':
    main()
