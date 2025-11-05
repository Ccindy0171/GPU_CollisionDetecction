"""
空间分割数据结构模块

该模块实现了Uniform Grid空间分割结构，用于加速最近邻查找。
使用GPU并行算法构建和查询网格。

Classes:
    UniformGrid: 均匀网格空间分割结构
"""

import cupy as cp
import numpy as np
from typing import Tuple, Optional


class UniformGrid:
    """
    GPU上的均匀网格空间分割结构
    
    将3D空间划分为规则的立方体网格单元，用于加速碰撞检测中的
    最近邻查找。使用空间哈希将3D坐标映射到1D索引。
    
    Attributes:
        world_min (cp.ndarray): 世界空间最小边界 [3]
        world_max (cp.ndarray): 世界空间最大边界 [3]
        cell_size (float): 网格单元大小
        resolution (cp.ndarray): 网格分辨率 [3]
        total_cells (int): 总单元数
        cell_starts (cp.ndarray): 每个单元的起始索引 [total_cells]
        cell_ends (cp.ndarray): 每个单元的结束索引 [total_cells]
    """
    
    def __init__(
        self,
        world_min: Tuple[float, float, float],
        world_max: Tuple[float, float, float],
        cell_size: float,
        device_id: int = 0
    ):
        """
        初始化均匀网格
        
        Args:
            world_min: 世界空间最小坐标 (x, y, z)
            world_max: 世界空间最大坐标 (x, y, z)
            cell_size: 网格单元大小，建议为平均物体直径的2倍
            device_id: GPU设备ID
            
        Raises:
            ValueError: 当参数无效时抛出
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        
        self.device_id = device_id
        
        with cp.cuda.Device(device_id):
            self.world_min = cp.array(world_min, dtype=cp.float32)
            self.world_max = cp.array(world_max, dtype=cp.float32)
            
            # 验证边界
            if cp.any(self.world_max <= self.world_min):
                raise ValueError(
                    "world_max must be greater than world_min in all dimensions"
                )
            
            self.cell_size = float(cell_size)
            
            # 计算网格分辨率
            world_size = self.world_max - self.world_min
            self.resolution = cp.ceil(world_size / cell_size).astype(cp.int32)
            self.total_cells = int(cp.prod(self.resolution))
            
            # 限制最大网格数（避免显存溢出）
            max_cells = 1000000  # RTX 3050显存限制
            if self.total_cells > max_cells:
                raise ValueError(
                    f"Grid resolution too high: {self.total_cells} cells. "
                    f"Maximum allowed: {max_cells}. "
                    f"Consider increasing cell_size."
                )
            
            # 网格数据结构
            self.cell_starts = cp.full(self.total_cells, -1, dtype=cp.int32)
            self.cell_ends = cp.full(self.total_cells, -1, dtype=cp.int32)
            
            # 排序缓冲区（延迟分配）
            self.sorted_indices: Optional[cp.ndarray] = None
            self.sorted_hashes: Optional[cp.ndarray] = None
    
    def get_grid_coord(self, positions: cp.ndarray) -> cp.ndarray:
        """
        将世界坐标转换为网格坐标
        
        Args:
            positions: 世界坐标 [N, 3]
        
        Returns:
            网格坐标 [N, 3]，整数类型
        """
        normalized = (positions - self.world_min) / self.cell_size
        grid_coords = cp.floor(normalized).astype(cp.int32)
        
        # 边界夹紧（确保所有坐标在有效范围内）
        grid_coords = cp.clip(grid_coords, 0, self.resolution - 1)
        return grid_coords
    
    def get_grid_hash(self, grid_coords: cp.ndarray) -> cp.ndarray:
        """
        将3D网格坐标转换为1D哈希值
        
        使用行优先顺序：hash = z * (res_y * res_x) + y * res_x + x
        
        Args:
            grid_coords: 网格坐标 [N, 3]
        
        Returns:
            1D哈希值 [N]
        """
        return (
            grid_coords[:, 2] * self.resolution[1] * self.resolution[0] +
            grid_coords[:, 1] * self.resolution[0] +
            grid_coords[:, 0]
        )
    
    def hash_to_coord(self, hashes: cp.ndarray) -> cp.ndarray:
        """
        将1D哈希值转换回3D网格坐标
        
        Args:
            hashes: 1D哈希值 [N]
        
        Returns:
            网格坐标 [N, 3]
        """
        z = hashes // (self.resolution[1] * self.resolution[0])
        remainder = hashes % (self.resolution[1] * self.resolution[0])
        y = remainder // self.resolution[0]
        x = remainder % self.resolution[0]
        
        return cp.stack([x, y, z], axis=1).astype(cp.int32)
    
    def clear(self) -> None:
        """清空网格数据"""
        self.cell_starts.fill(-1)
        self.cell_ends.fill(-1)
    
    def get_neighbor_cells(
        self,
        grid_coord: Tuple[int, int, int],
        radius: int = 1
    ) -> cp.ndarray:
        """
        获取指定网格单元周围的邻居单元哈希值
        
        Args:
            grid_coord: 中心网格坐标 (x, y, z)
            radius: 搜索半径（网格单元数），默认为1（3x3x3）
        
        Returns:
            邻居单元的哈希值数组
        """
        x, y, z = grid_coord
        neighbors = []
        
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # 边界检查
                    if (0 <= nx < self.resolution[0] and
                        0 <= ny < self.resolution[1] and
                        0 <= nz < self.resolution[2]):
                        
                        hash_val = (nz * self.resolution[1] * self.resolution[0] +
                                   ny * self.resolution[0] + nx)
                        neighbors.append(hash_val)
        
        return cp.array(neighbors, dtype=cp.int32)
    
    def get_cell_occupancy(self) -> Tuple[int, int, float]:
        """
        获取网格占用统计信息
        
        Returns:
            (已占用单元数, 总单元数, 占用率)
        """
        occupied = cp.sum(self.cell_starts >= 0).item()
        total = self.total_cells
        occupancy = occupied / total if total > 0 else 0.0
        
        return occupied, total, occupancy
    
    def get_cell_distribution(self) -> Dict[str, float]:
        """
        获取单元内物体数量的分布统计
        
        Returns:
            包含统计信息的字典
        """
        # 计算每个单元的物体数量
        cell_counts = self.cell_ends - self.cell_starts
        cell_counts = cell_counts[cell_counts >= 0]  # 只统计非空单元
        
        if len(cell_counts) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0,
                'median': 0.0
            }
        
        return {
            'mean': float(cp.mean(cell_counts)),
            'std': float(cp.std(cell_counts)),
            'min': int(cp.min(cell_counts)),
            'max': int(cp.max(cell_counts)),
            'median': float(cp.median(cell_counts))
        }
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"UniformGrid(resolution={tuple(cp.asnumpy(self.resolution))}, "
            f"cell_size={self.cell_size:.2f}, "
            f"total_cells={self.total_cells})"
        )


# 导入字典类型
from typing import Dict
