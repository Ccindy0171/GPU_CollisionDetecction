#!/usr/bin/env python3
"""
Helper function to generate non-overlapping initial positions
"""

import numpy as np
from typing import Tuple


def generate_non_overlapping_positions(
    num_objects: int,
    radii: np.ndarray,
    world_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    max_attempts: int = 100
) -> np.ndarray:
    """
    生成不重叠的初始位置
    
    使用网格方法放置物体，确保没有初始重叠
    
    Args:
        num_objects: 物体数量
        radii: 物体半径数组 [N]
        world_bounds: 世界边界 ((xmin, ymin, zmin), (xmax, ymax, zmax))
        max_attempts: 每个物体的最大尝试次数
        
    Returns:
        positions: 物体位置 [N, 3]
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = world_bounds
    
    # 按半径从大到小排序，先放置大球
    sorted_indices = np.argsort(radii)[::-1]
    positions = np.zeros((num_objects, 3), dtype=np.float32)
    
    # 使用泊松盘采样的简化版本
    placed_count = 0
    
    # 分层放置
    y_layers = np.linspace(ymax * 0.6, ymax * 0.9, 4)  # 4层
    
    for idx in sorted_indices:
        r = radii[idx]
        placed = False
        
        # 选择一层
        layer_y = y_layers[placed_count % len(y_layers)]
        
        for attempt in range(max_attempts):
            # 随机生成候选位置
            pos = np.array([
                np.random.uniform(xmin + r, xmax - r),
                layer_y + np.random.uniform(-2, 2),
                np.random.uniform(zmin + r, zmax - r)
            ], dtype=np.float32)
            
            # 检查是否与已放置的物体重叠
            if placed_count == 0:
                # 第一个物体，直接放置
                positions[idx] = pos
                placed = True
                break
            
            # 计算与所有已放置物体的距离
            placed_positions = positions[sorted_indices[:placed_count]]
            placed_radii = radii[sorted_indices[:placed_count]]
            
            distances = np.linalg.norm(placed_positions - pos, axis=1)
            min_distances = placed_radii + r + 0.1  # 添加0.1m的安全间距
            
            # 如果没有重叠，放置此物体
            if np.all(distances >= min_distances):
                positions[idx] = pos
                placed = True
                break
        
        if not placed:
            # 如果无法找到不重叠的位置，放在高处
            positions[idx] = np.array([
                np.random.uniform(xmin + r, xmax - r),
                ymax * 0.95,
                np.random.uniform(zmin + r, zmax - r)
            ], dtype=np.float32)
            print(f"Warning: Object {idx} placed without overlap check (attempt limit reached)")
        
        placed_count += 1
    
    return positions


def verify_no_overlaps(positions: np.ndarray, radii: np.ndarray) -> Tuple[int, float]:
    """
    验证没有重叠
    
    Returns:
        (num_overlaps, max_penetration)
    """
    num_objects = len(positions)
    num_overlaps = 0
    max_penetration = 0.0
    
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            dist = np.linalg.norm(positions[j] - positions[i])
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                num_overlaps += 1
                penetration = min_dist - dist
                max_penetration = max(max_penetration, penetration)
    
    return num_overlaps, max_penetration
