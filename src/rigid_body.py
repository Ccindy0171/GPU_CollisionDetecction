"""
刚体物理系统模块

该模块定义了GPU上的刚体物理系统，包含所有物体的物理属性。
使用结构化数组（SOA - Structure of Arrays）布局以优化GPU内存访问。

Classes:
    RigidBodySystem: 刚体物理系统的主类
"""

import cupy as cp
import numpy as np
from typing import Dict, Tuple, Optional


class RigidBodySystem:
    """
    GPU上的刚体物理系统
    
    使用SOA（Structure of Arrays）数据布局来优化GPU内存访问模式。
    所有物理属性都存储在GPU显存中，减少CPU-GPU数据传输。
    
    Attributes:
        num_objects (int): 物体数量
        device_id (int): GPU设备ID
        positions (cp.ndarray): 物体位置 [N, 3]
        velocities (cp.ndarray): 物体速度 [N, 3]
        accelerations (cp.ndarray): 物体加速度 [N, 3]
        forces (cp.ndarray): 累积力 [N, 3]
        radii (cp.ndarray): 物体半径 [N]
        masses (cp.ndarray): 物体质量 [N]
        restitutions (cp.ndarray): 弹性系数 [N]
        frictions (cp.ndarray): 摩擦系数 [N]
        colors (cp.ndarray): 渲染颜色 [N, 3]
    """
    
    def __init__(self, num_objects: int, device_id: int = 0):
        """
        初始化刚体系统
        
        Args:
            num_objects: 物体数量，建议RTX 3050使用5000-15000范围
            device_id: GPU设备ID，默认为0
            
        Raises:
            ValueError: 当num_objects <= 0时抛出
            RuntimeError: 当CUDA设备不可用时抛出
        """
        if num_objects <= 0:
            raise ValueError(f"num_objects must be positive, got {num_objects}")
        
        self.num_objects = num_objects
        self.device_id = device_id
        
        try:
            with cp.cuda.Device(device_id):
                # 动力学属性 - 使用float32以节省显存并提高性能
                self.positions = cp.zeros((num_objects, 3), dtype=cp.float32)
                self.velocities = cp.zeros((num_objects, 3), dtype=cp.float32)
                self.accelerations = cp.zeros((num_objects, 3), dtype=cp.float32)
                self.forces = cp.zeros((num_objects, 3), dtype=cp.float32)
                
                # 物理属性
                self.radii = cp.ones(num_objects, dtype=cp.float32) * 0.5
                self.masses = cp.ones(num_objects, dtype=cp.float32) * 1.0
                self.restitutions = cp.ones(num_objects, dtype=cp.float32) * 0.8
                self.frictions = cp.ones(num_objects, dtype=cp.float32) * 0.3
                
                # 网格索引（用于空间分割）
                self.grid_indices = cp.zeros(num_objects, dtype=cp.int32)
                
                # 状态标记
                self.is_active = cp.ones(num_objects, dtype=cp.bool_)
                
                # 渲染属性
                self.colors = cp.random.rand(num_objects, 3).astype(cp.float32)
                
        except cp.cuda.runtime.CUDARuntimeError as e:
            raise RuntimeError(f"Failed to initialize CUDA device {device_id}: {e}")
    
    def set_positions(self, positions: np.ndarray) -> None:
        """
        设置物体位置
        
        Args:
            positions: NumPy数组 [N, 3] 或可以广播到该形状的数组
            
        Raises:
            ValueError: 当positions形状不兼容时抛出
        """
        if positions.shape[0] != self.num_objects:
            raise ValueError(
                f"positions first dimension must be {self.num_objects}, "
                f"got {positions.shape[0]}"
            )
        self.positions[:] = cp.asarray(positions, dtype=cp.float32)
    
    def set_velocities(self, velocities: np.ndarray) -> None:
        """
        设置物体速度
        
        Args:
            velocities: NumPy数组 [N, 3] 或可以广播到该形状的数组
            
        Raises:
            ValueError: 当velocities形状不兼容时抛出
        """
        if velocities.shape[0] != self.num_objects:
            raise ValueError(
                f"velocities first dimension must be {self.num_objects}, "
                f"got {velocities.shape[0]}"
            )
        self.velocities[:] = cp.asarray(velocities, dtype=cp.float32)
    
    def set_radii(self, radii: np.ndarray) -> None:
        """
        设置物体半径
        
        Args:
            radii: NumPy数组 [N] 或标量
            
        Raises:
            ValueError: 当radii包含非正值时抛出
        """
        radii_array = cp.asarray(radii, dtype=cp.float32)
        if cp.any(radii_array <= 0):
            raise ValueError("All radii must be positive")
        self.radii[:] = radii_array
    
    def set_masses(self, masses: np.ndarray) -> None:
        """
        设置物体质量
        
        Args:
            masses: NumPy数组 [N] 或标量
            
        Raises:
            ValueError: 当masses包含非正值时抛出
        """
        masses_array = cp.asarray(masses, dtype=cp.float32)
        if cp.any(masses_array <= 0):
            raise ValueError("All masses must be positive")
        self.masses[:] = masses_array
    
    def set_restitutions(self, restitutions: np.ndarray) -> None:
        """
        设置弹性系数
        
        Args:
            restitutions: NumPy数组 [N] 或标量，范围应在[0, 1]
            
        Raises:
            ValueError: 当restitutions不在有效范围时抛出
        """
        rest_array = cp.asarray(restitutions, dtype=cp.float32)
        if cp.any((rest_array < 0) | (rest_array > 1)):
            raise ValueError("Restitutions must be in range [0, 1]")
        self.restitutions[:] = rest_array
    
    def set_colors(self, colors: np.ndarray) -> None:
        """
        设置渲染颜色
        
        Args:
            colors: NumPy数组 [N, 3]，RGB值范围应在[0, 1]
        """
        self.colors[:] = cp.asarray(colors, dtype=cp.float32)
    
    def to_cpu(self) -> Dict[str, np.ndarray]:
        """
        将GPU数据传输到CPU
        
        用于渲染、分析或数据导出。由于CPU-GPU传输较慢，
        应该尽量减少调用频率。
        
        Returns:
            包含所有物理属性的字典，值为NumPy数组
        """
        return {
            'positions': cp.asnumpy(self.positions),
            'velocities': cp.asnumpy(self.velocities),
            'radii': cp.asnumpy(self.radii),
            'masses': cp.asnumpy(self.masses),
            'colors': cp.asnumpy(self.colors),
            'restitutions': cp.asnumpy(self.restitutions),
        }
    
    def get_kinetic_energy(self) -> float:
        """
        计算系统总动能
        
        Returns:
            系统总动能（标量）
        """
        v_squared = cp.sum(self.velocities ** 2, axis=1)
        ke = 0.5 * cp.sum(self.masses * v_squared)
        return float(ke)
    
    def get_momentum(self) -> np.ndarray:
        """
        计算系统总动量
        
        Returns:
            系统总动量向量 [3]
        """
        momentum = cp.sum(self.masses[:, None] * self.velocities, axis=0)
        return cp.asnumpy(momentum)
    
    def reset_forces(self) -> None:
        """重置所有累积力为零"""
        self.forces.fill(0.0)
    
    def apply_force(self, forces: cp.ndarray) -> None:
        """
        应用外力到物体
        
        Args:
            forces: CuPy数组 [N, 3]，要应用的力
        """
        self.forces += forces
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"RigidBodySystem(num_objects={self.num_objects}, "
            f"device_id={self.device_id})"
        )
    
    def __len__(self) -> int:
        """返回物体数量"""
        return self.num_objects
