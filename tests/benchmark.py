#!/usr/bin/env python3
"""
性能测试脚本

测试不同规模下的GPU碰撞检测性能，生成性能分析报告。

运行方式:
    python tests/benchmark.py

输出:
    - output/benchmark_scaling.png 规模性能图
    - output/benchmark_results.csv 详细测试数据
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from src import PhysicsSimulator, PerformanceMonitor
import time


def benchmark_scaling():
    """测试不同物体数量下的性能"""
    print("=" * 70)
    print("GPU Collision Detection - Performance Benchmark")
    print("=" * 70)
    
    # 测试规模（针对RTX 3050调整）
    scales = [1000, 2000, 5000, 8000, 10000, 15000]
    world_bounds = ((-50, -50, -50), (50, 50, 50))
    cell_size = 2.5
    
    results = []
    
    for num_objects in scales:
        print(f"\n{'='*70}")
        print(f"Testing with {num_objects} objects...")
        print(f"{'='*70}")
        
        try:
            # 初始化仿真器
            sim = PhysicsSimulator(
                num_objects,
                world_bounds,
                cell_size=cell_size,
                dt=1.0/60.0
            )
            
            # 随机初始化
            with cp.cuda.Device(sim.device_id):
                # 位置
                positions = np.random.uniform(-40, 40, (num_objects, 3)).astype(np.float32)
                sim.bodies.set_positions(positions)
                
                # 速度
                velocities = np.random.uniform(-5, 5, (num_objects, 3)).astype(np.float32)
                sim.bodies.set_velocities(velocities)
                
                # 半径
                radii = np.random.uniform(0.3, 0.7, num_objects).astype(np.float32)
                sim.bodies.set_radii(radii)
                
                # 质量
                masses = (radii ** 3 * 1000.0).astype(np.float32)
                sim.bodies.set_masses(masses)
                
                # 弹性系数
                restitutions = np.random.uniform(0.7, 0.9, num_objects).astype(np.float32)
                sim.bodies.set_restitutions(restitutions)
            
            # 预热（JIT编译等）
            print("  Warming up...")
            for _ in range(10):
                sim.step()
            
            # 测试
            print("  Running benchmark...")
            times = []
            collision_counts = []
            
            num_test_frames = 100
            for i in range(num_test_frames):
                step_info = sim.step()
                times.append(step_info['total_time'])
                collision_counts.append(step_info['num_collisions'])
                
                if (i + 1) % 20 == 0:
                    print(f"    Progress: {i+1}/{num_test_frames} frames")
            
            # 统计
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1000.0 / avg_time
            avg_collisions = np.mean(collision_counts)
            
            # 网格信息
            sys_info = sim.get_system_info()
            
            result = {
                'num_objects': num_objects,
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'fps': fps,
                'avg_collisions': avg_collisions,
                'grid_resolution': sys_info['grid_resolution'],
                'total_cells': sys_info['total_cells'],
                'cell_occupancy': sys_info['cell_occupancy'],
            }
            
            results.append(result)
            
            # 输出结果
            print(f"\n  Results:")
            print(f"    Avg Time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"    FPS: {fps:.1f}")
            print(f"    Min/Max Time: {min_time:.2f} / {max_time:.2f} ms")
            print(f"    Avg Collisions: {avg_collisions:.0f}")
            print(f"    Grid Occupancy: {sys_info['cell_occupancy']*100:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'num_objects': num_objects,
                'error': str(e)
            })
    
    return results


def plot_results(results, output_dir):
    """绘制性能结果图表"""
    # 过滤掉失败的测试
    valid_results = [r for r in results if 'avg_time_ms' in r]
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    nums = [r['num_objects'] for r in valid_results]
    times = [r['avg_time_ms'] for r in valid_results]
    fps_list = [r['fps'] for r in valid_results]
    collisions = [r['avg_collisions'] for r in valid_results]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 帧时间 vs 物体数量
    ax = axes[0, 0]
    ax.plot(nums, times, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Objects')
    ax.set_ylabel('Frame Time (ms)')
    ax.set_title('Frame Time vs Number of Objects')
    ax.grid(True, alpha=0.3)
    
    # 添加目标线（16.67ms = 60fps）
    ax.axhline(y=16.67, color='r', linestyle='--', label='60 FPS Target')
    ax.legend()
    
    # 2. FPS vs 物体数量
    ax = axes[0, 1]
    ax.plot(nums, fps_list, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Objects')
    ax.set_ylabel('FPS')
    ax.set_title('FPS vs Number of Objects')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=60, color='r', linestyle='--', label='60 FPS Target')
    ax.legend()
    
    # 3. 碰撞数量 vs 物体数量
    ax = axes[1, 0]
    ax.plot(nums, collisions, 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Objects')
    ax.set_ylabel('Average Collisions per Frame')
    ax.set_title('Collisions vs Number of Objects')
    ax.grid(True, alpha=0.3)
    
    # 4. 性能分解
    ax = axes[1, 1]
    # 这里需要额外数据，暂时显示总结
    summary_text = "Performance Summary\n\n"
    for r in valid_results:
        summary_text += f"N={r['num_objects']:5d}: {r['fps']:5.1f} FPS\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    
    # 保存
    plot_path = os.path.join(output_dir, 'benchmark_scaling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark plot saved to: {plot_path}")
    plt.close()  # Close figure instead of showing


def save_results_csv(results, output_dir):
    """保存结果到CSV"""
    import csv
    
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    
    with open(csv_path, 'w', newline='') as f:
        if len(results) == 0:
            return
        
        # 获取所有可能的字段
        fieldnames = list(results[0].keys())
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {csv_path}")


def main():
    """主函数"""
    # 运行基准测试
    results = benchmark_scaling()
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制结果
    plot_results(results, output_dir)
    
    # 保存CSV
    save_results_csv(results, output_dir)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    
    valid_results = [r for r in results if 'avg_time_ms' in r]
    
    if len(valid_results) > 0:
        print("\nPerformance Results:")
        print(f"{'Objects':>10} {'Avg Time':>12} {'FPS':>8} {'Collisions':>12}")
        print("-" * 50)
        for r in valid_results:
            print(f"{r['num_objects']:10d} "
                  f"{r['avg_time_ms']:10.2f} ms "
                  f"{r['fps']:7.1f} "
                  f"{r['avg_collisions']:11.0f}")
        
        # 找到最佳配置
        best_fps = max(valid_results, key=lambda x: x['fps'])
        print(f"\nBest FPS Configuration:")
        print(f"  Objects: {best_fps['num_objects']}")
        print(f"  FPS: {best_fps['fps']:.1f}")
        print(f"  Frame Time: {best_fps['avg_time_ms']:.2f} ms")
        
        # 推荐配置（接近60fps的最大物体数）
        near_60fps = [r for r in valid_results if r['fps'] >= 55]
        if near_60fps:
            recommended = max(near_60fps, key=lambda x: x['num_objects'])
            print(f"\nRecommended Configuration for 60 FPS:")
            print(f"  Objects: {recommended['num_objects']}")
            print(f"  FPS: {recommended['fps']:.1f}")
            print(f"  Frame Time: {recommended['avg_time_ms']:.2f} ms")
    
    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
