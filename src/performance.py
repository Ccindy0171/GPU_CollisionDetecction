"""
性能分析模块

该模块提供性能监控和基准测试工具。

Classes:
    PerformanceMonitor: 性能监控器
"""

import numpy as np
import cupy as cp
from collections import defaultdict
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import time


class PerformanceMonitor:
    """
    性能监控工具
    
    记录和分析仿真过程中的性能指标。
    
    Attributes:
        metrics: 存储各类指标的字典
        events: CUDA事件字典（用于GPU计时）
    """
    
    def __init__(self):
        """初始化性能监控器"""
        self.metrics = defaultdict(list)
        self.events = {}
        self.cpu_times = {}
    
    def start_event(self, name: str, use_gpu: bool = True) -> None:
        """
        开始计时事件
        
        Args:
            name: 事件名称
            use_gpu: 是否使用GPU事件（True）或CPU时间（False）
        """
        if use_gpu:
            event = cp.cuda.Event()
            event.record()
            self.events[name] = event
        else:
            self.cpu_times[name] = time.perf_counter()
    
    def end_event(self, name: str, use_gpu: bool = True) -> float:
        """
        结束计时事件并记录
        
        Args:
            name: 事件名称
            use_gpu: 是否使用GPU事件（True）或CPU时间（False）
            
        Returns:
            经过的时间（毫秒）
        """
        if use_gpu:
            if name not in self.events:
                return 0.0
            
            end_event = cp.cuda.Event()
            end_event.record()
            end_event.synchronize()
            
            elapsed = cp.cuda.get_elapsed_time(self.events[name], end_event)
            self.metrics[name].append(elapsed)
            del self.events[name]
            
        else:
            if name not in self.cpu_times:
                return 0.0
            
            elapsed = (time.perf_counter() - self.cpu_times[name]) * 1000.0
            self.metrics[name].append(elapsed)
            del self.cpu_times[name]
        
        return elapsed
    
    def record_metric(self, name: str, value: float) -> None:
        """
        记录自定义指标
        
        Args:
            name: 指标名称
            value: 指标值
        """
        self.metrics[name].append(value)
    
    def get_statistics(self, recent_frames: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        获取统计数据
        
        Args:
            recent_frames: 只统计最近N帧，None表示统计所有帧
        
        Returns:
            包含每个指标统计信息的字典
        """
        stats = {}
        
        for name, values in self.metrics.items():
            if len(values) == 0:
                continue
            
            # 选择统计范围
            if recent_frames is not None and len(values) > recent_frames:
                values_array = np.array(values[-recent_frames:])
            else:
                values_array = np.array(values)
            
            stats[name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'count': len(values_array)
            }
        
        return stats
    
    def print_statistics(self, recent_frames: Optional[int] = None) -> None:
        """
        打印统计信息
        
        Args:
            recent_frames: 只统计最近N帧
        """
        stats = self.get_statistics(recent_frames)
        
        print("=" * 70)
        print("Performance Statistics")
        if recent_frames:
            print(f"(Last {recent_frames} frames)")
        print("=" * 70)
        
        for name, stat in stats.items():
            print(f"\n{name}:")
            print(f"  Mean:   {stat['mean']:8.2f} ms")
            print(f"  Std:    {stat['std']:8.2f} ms")
            print(f"  Min:    {stat['min']:8.2f} ms")
            print(f"  Max:    {stat['max']:8.2f} ms")
            print(f"  Median: {stat['median']:8.2f} ms")
        
        print("=" * 70)
    
    def plot_timeline(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 10)
    ) -> None:
        """
        绘制性能时间线图
        
        Args:
            metrics: 要绘制的指标列表，None表示绘制所有指标
            save_path: 保存路径，None表示只显示不保存
            figsize: 图形大小
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        # 过滤出有数据的指标
        metrics = [m for m in metrics if m in self.metrics and len(self.metrics[m]) > 0]
        
        if len(metrics) == 0:
            print("No metrics to plot")
            return
        
        # 创建子图
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        # 绘制每个指标
        for ax, metric in zip(axes, metrics):
            values = self.metrics[metric]
            frames = range(len(values))
            
            ax.plot(frames, values, linewidth=0.5, alpha=0.7)
            
            # 添加移动平均线
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(values)), moving_avg, 
                       'r-', linewidth=2, label=f'Moving Avg (window={window})')
                ax.legend()
            
            ax.set_ylabel(f'{metric} (ms)')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{metric}')
        
        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Timeline plot saved to {save_path}")
        
        plt.show()
    
    def plot_distribution(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 6)
    ) -> None:
        """
        绘制指标分布直方图
        
        Args:
            metrics: 要绘制的指标列表
            save_path: 保存路径
            figsize: 图形大小
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
        
        metrics = [m for m in metrics if m in self.metrics and len(self.metrics[m]) > 0]
        
        if len(metrics) == 0:
            print("No metrics to plot")
            return
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            values = self.metrics[metric]
            ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{metric} (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def export_to_csv(self, filename: str) -> None:
        """
        导出数据到CSV文件
        
        Args:
            filename: 输出文件名
        """
        import csv
        
        # 找到最长的序列长度
        max_len = max(len(values) for values in self.metrics.values())
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow(['Frame'] + list(self.metrics.keys()))
            
            # 写入数据
            for i in range(max_len):
                row = [i]
                for metric_name in self.metrics.keys():
                    values = self.metrics[metric_name]
                    if i < len(values):
                        row.append(values[i])
                    else:
                        row.append('')
                writer.writerow(row)
        
        print(f"Data exported to {filename}")
    
    def clear(self) -> None:
        """清空所有记录的数据"""
        self.metrics.clear()
        self.events.clear()
        self.cpu_times.clear()
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        total_frames = max(len(v) for v in self.metrics.values()) if self.metrics else 0
        return (
            f"PerformanceMonitor(metrics={len(self.metrics)}, "
            f"frames={total_frames})"
        )
