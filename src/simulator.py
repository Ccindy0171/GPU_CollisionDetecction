"""
物理仿真器模块

该模块实现了完整的GPU加速物理仿真系统，整合了空间分割、
碰撞检测和物理积分等功能。

Classes:
    PhysicsSimulator: 主物理仿真器类
"""

import cupy as cp
import numpy as np
from typing import Dict, Tuple, Optional
import time

from .rigid_body import RigidBodySystem
from .spatial_grid import UniformGrid
from .kernels import (
    COMPUTE_GRID_HASH_KERNEL,
    FIND_CELL_START_KERNEL,
    BROAD_PHASE_KERNEL,
    COLLISION_RESPONSE_KERNEL,
    INTEGRATE_KERNEL,
)


class PhysicsSimulator:
    """
    基于GPU的物理仿真器
    
    整合了空间分割、碰撞检测和物理积分的完整仿真流程。
    针对RTX 3050显卡优化（建议物体数量5000-15000）。
    
    Attributes:
        bodies (RigidBodySystem): 刚体系统
        grid (UniformGrid): 空间网格
        gravity (cp.ndarray): 重力加速度向量
        dt (float): 时间步长
        damping (float): 速度阻尼系数
    """
    
    def __init__(
        self,
        num_objects: int,
        world_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        cell_size: float = 2.0,
        device_id: int = 0,
        dt: float = 1.0 / 60.0,
        gravity: Tuple[float, float, float] = (0.0, -9.8, 0.0),
        damping: float = 0.01
    ):
        """
        初始化物理仿真器
        
        Args:
            num_objects: 物体数量（建议5000-15000 for RTX 3050）
            world_bounds: 世界边界 ((xmin, ymin, zmin), (xmax, ymax, zmax))
            cell_size: 网格单元大小（建议为平均物体直径的2倍）
            device_id: GPU设备ID
            dt: 时间步长（秒）
            gravity: 重力加速度 (x, y, z)
            damping: 速度阻尼系数 [0, 1]
            
        Raises:
            ValueError: 当参数无效时抛出
            RuntimeError: 当GPU资源不足时抛出
        """
        if num_objects <= 0:
            raise ValueError(f"num_objects must be positive, got {num_objects}")
        
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        if not 0 <= damping <= 1:
            raise ValueError(f"damping must be in [0, 1], got {damping}")
        
        self.device_id = device_id
        self.num_objects = num_objects
        
        try:
            with cp.cuda.Device(device_id):
                # 初始化刚体系统
                self.bodies = RigidBodySystem(num_objects, device_id)
                
                # 初始化空间网格
                self.grid = UniformGrid(
                    world_bounds[0],
                    world_bounds[1],
                    cell_size,
                    device_id
                )
                
                # 物理参数
                self.gravity = cp.array(gravity, dtype=cp.float32)
                self.dt = float(dt)
                self.damping = float(damping)
                
                # 碰撞对缓冲区（估计每个物体平均20个潜在碰撞）
                self.max_pairs = min(num_objects * 20, 500000)  # 限制最大值
                self.collision_pairs = cp.zeros((self.max_pairs, 2), dtype=cp.int32)
                self.pair_count = cp.zeros(1, dtype=cp.int32)
                
                # 排序缓冲区
                self.sorted_indices = cp.zeros(num_objects, dtype=cp.int32)
                self.grid_hashes = cp.zeros(num_objects, dtype=cp.int32)
                
                # 性能统计
                self.stats = {
                    'grid_build_time': 0.0,
                    'collision_detect_time': 0.0,
                    'collision_resolve_time': 0.0,
                    'integrate_time': 0.0,
                    'total_time': 0.0,
                }
                
                # 帧计数
                self.frame_count = 0
                
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(f"Failed to initialize simulator: {e}")
    
    def build_grid(self) -> None:
        """
        构建空间网格
        
        执行步骤：
        1. 计算每个物体的网格哈希
        2. 按哈希值排序物体索引
        3. 找到每个网格单元的起始和结束位置
        """
        with cp.cuda.Device(self.device_id):
            # 清空网格
            self.grid.clear()
            
            # 1. 计算网格哈希
            threads_per_block = 256
            blocks = (self.num_objects + threads_per_block - 1) // threads_per_block
            
            COMPUTE_GRID_HASH_KERNEL(
                (blocks,), (threads_per_block,),
                (
                    self.bodies.positions,
                    self.grid_hashes,
                    self.grid.world_min,
                    np.float32(self.grid.cell_size),
                    self.grid.resolution,
                    self.num_objects
                )
            )
            
            # 2. 按哈希值排序（CuPy的argsort在GPU上执行）
            self.sorted_indices = cp.argsort(self.grid_hashes).astype(cp.int32)
            sorted_hashes = self.grid_hashes[self.sorted_indices]
            
            # 3. 找到每个单元的起始和结束位置
            FIND_CELL_START_KERNEL(
                (blocks,), (threads_per_block,),
                (
                    sorted_hashes,
                    self.grid.cell_starts,
                    self.grid.cell_ends,
                    self.num_objects
                )
            )
    
    def detect_collisions(self) -> int:
        """
        检测碰撞（Broad Phase）
        
        遍历每个物体，检查其周围27个网格单元中的邻居物体。
        
        Returns:
            检测到的碰撞对数量
        """
        with cp.cuda.Device(self.device_id):
            # 重置碰撞对计数
            self.pair_count.fill(0)
            
            # 执行Broad Phase碰撞检测
            threads_per_block = 256
            blocks = (self.num_objects + threads_per_block - 1) // threads_per_block
            
            BROAD_PHASE_KERNEL(
                (blocks,), (threads_per_block,),
                (
                    self.bodies.positions,
                    self.bodies.radii,
                    self.grid.cell_starts,
                    self.grid.cell_ends,
                    self.sorted_indices,
                    self.grid.resolution,
                    np.float32(self.grid.cell_size),
                    self.grid.world_min,
                    self.collision_pairs,
                    self.pair_count,
                    self.num_objects,
                    self.max_pairs
                )
            )
            
            num_pairs = int(self.pair_count[0])
            
            # 检查是否溢出
            if num_pairs >= self.max_pairs:
                print(f"Warning: Collision pair buffer overflow! "
                      f"Detected {num_pairs}, buffer size {self.max_pairs}")
                num_pairs = self.max_pairs
            
            return num_pairs
    
    def resolve_collisions(self, num_pairs: int) -> None:
        """
        解决碰撞（应用冲量法）
        
        Args:
            num_pairs: 碰撞对数量
        """
        if num_pairs == 0:
            return
        
        with cp.cuda.Device(self.device_id):
            threads_per_block = 256
            blocks = (num_pairs + threads_per_block - 1) // threads_per_block
            
            COLLISION_RESPONSE_KERNEL(
                (blocks,), (threads_per_block,),
                (
                    self.bodies.positions,
                    self.bodies.velocities,
                    self.bodies.radii,
                    self.bodies.masses,
                    self.bodies.restitutions,
                    self.collision_pairs,
                    num_pairs
                )
            )
    
    def integrate(self) -> None:
        """
        物理积分（半隐式Euler方法）
        
        更新速度和位置，处理边界碰撞。
        """
        with cp.cuda.Device(self.device_id):
            threads_per_block = 256
            blocks = (self.num_objects + threads_per_block - 1) // threads_per_block
            
            INTEGRATE_KERNEL(
                (blocks,), (threads_per_block,),
                (
                    self.bodies.positions,
                    self.bodies.velocities,
                    self.bodies.forces,
                    self.bodies.masses,
                    self.gravity,
                    np.float32(self.dt),      # Explicit cast to np.float32
                    np.float32(self.damping), # Explicit cast to np.float32
                    self.grid.world_min,
                    self.grid.world_max,
                    self.bodies.radii,
                    self.bodies.restitutions,
                    self.num_objects
                )
            )
            
            # 清空力累积
            self.bodies.reset_forces()
    
    def step(self) -> Dict[str, any]:
        """
        执行一个仿真步
        
        完整流程：
        1. 构建空间网格
        2. 检测碰撞
        3. 解决碰撞
        4. 物理积分
        
        Returns:
            包含性能统计和碰撞信息的字典
        """
        with cp.cuda.Device(self.device_id):
            # 使用CUDA事件进行精确计时
            start_total = cp.cuda.Event()
            end_total = cp.cuda.Event()
            start_total.record()
            
            # 1. 物理积分（先更新位置和速度）
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            self.integrate()
            end.record()
            end.synchronize()
            self.stats['integrate_time'] = cp.cuda.get_elapsed_time(start, end)
            
            # 2. 构建网格（基于新位置）
            start.record()
            self.build_grid()
            end.record()
            end.synchronize()
            self.stats['grid_build_time'] = cp.cuda.get_elapsed_time(start, end)
            
            # 3. 检测并解决碰撞（多次迭代以完全分离重叠的物体）
            start.record()
            total_collisions = 0
            # 迭代多次，每次都重新构建网格并检测碰撞
            # 这样可以处理连锁碰撞和确保物体完全分离
            for iteration in range(5):  # 增加迭代次数以更彻底地解决碰撞
                num_pairs = self.detect_collisions()
                total_collisions += num_pairs
                if num_pairs > 0:
                    self.resolve_collisions(num_pairs)
                    # 在迭代中重新构建网格（除了最后一次）
                    if iteration < 4:
                        self.build_grid()
                else:
                    break  # 如果没有碰撞了，提前退出
            end.record()
            end.synchronize()
            collision_time = cp.cuda.get_elapsed_time(start, end)
            self.stats['collision_detect_time'] = collision_time * 0.5  # 估算检测时间
            self.stats['collision_resolve_time'] = collision_time * 0.5  # 估算响应时间
            
            # 总时间
            end_total.record()
            end_total.synchronize()
            self.stats['total_time'] = cp.cuda.get_elapsed_time(start_total, end_total)
            
            self.frame_count += 1
            
            return {
                'num_collisions': total_collisions,  # 返回所有迭代的总碰撞数
                'frame': self.frame_count,
                **self.stats
            }
    
    def get_stats(self) -> Dict[str, float]:
        """
        获取性能统计
        
        Returns:
            包含各阶段耗时和FPS的字典
        """
        total = self.stats['total_time']
        return {
            **self.stats,
            'fps': 1000.0 / total if total > 0 else 0.0,
        }
    
    def get_system_info(self) -> Dict[str, any]:
        """
        获取系统信息
        
        Returns:
            包含物体数量、网格信息等的字典
        """
        occupied, total, occupancy = self.grid.get_cell_occupancy()
        cell_dist = self.grid.get_cell_distribution()
        
        return {
            'num_objects': self.num_objects,
            'grid_resolution': tuple(cp.asnumpy(self.grid.resolution)),
            'total_cells': self.grid.total_cells,
            'occupied_cells': occupied,
            'cell_occupancy': occupancy,
            'cell_distribution': cell_dist,
            'kinetic_energy': self.bodies.get_kinetic_energy(),
            'momentum': self.bodies.get_momentum(),
            'frame_count': self.frame_count,
        }
    
    def reset(self) -> None:
        """重置仿真状态"""
        with cp.cuda.Device(self.device_id):
            self.bodies.velocities.fill(0)
            self.bodies.accelerations.fill(0)
            self.bodies.forces.fill(0)
            self.frame_count = 0
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"PhysicsSimulator(num_objects={self.num_objects}, "
            f"dt={self.dt:.4f}, frame={self.frame_count})"
        )
