"""
GPU碰撞检测系统
基于CuPy的高性能GPU加速碰撞检测与物理仿真

模块说明:
    - rigid_body: 刚体物理系统
    - spatial_grid: 空间分割数据结构
    - kernels: CUDA内核函数
    - simulator: 主仿真器
    - visualizer: 可视化工具
    - performance: 性能分析工具

作者: Cindy
日期: 2025-11-04
"""

__version__ = "1.0.0"
__author__ = "Cindy"

from .rigid_body import RigidBodySystem
from .spatial_grid import UniformGrid
from .simulator import PhysicsSimulator
from .visualizer import RealtimeVisualizer, VideoExporter
from .performance import PerformanceMonitor

__all__ = [
    'RigidBodySystem',
    'UniformGrid',
    'PhysicsSimulator',
    'RealtimeVisualizer',
    'VideoExporter',
    'PerformanceMonitor',
]
