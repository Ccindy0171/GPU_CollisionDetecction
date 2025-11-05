# GPU碰撞检测系统

基于GPU的高性能碰撞检测与物理仿真系统，使用CuPy实现。

## 项目概述

本项目实现了一套完整的GPU加速碰撞检测系统，包括：

1. **基于Uniform Grid的最近邻查找算法** - 实现O(N)复杂度的大规模碰撞检测
2. **完整的物理仿真引擎** - 包含刚体动力学、碰撞响应和边界处理
3. **性能测试与分析工具** - 详细的性能监控和基准测试
4. **可视化和动画导出** - 实时3D可视化和高质量视频导出

## 系统要求

### 硬件要求
- NVIDIA GPU (支持CUDA)
- 测试环境：RTX 3050 (4GB VRAM)
- 推荐：8GB+ 系统内存

### 软件要求
- Python 3.8+
- CUDA 11.x 或 12.x
- Linux / Windows

## 安装指南

### 1. 安装CUDA

确保已安装NVIDIA驱动和CUDA Toolkit。检查CUDA版本：

```bash
nvcc --version
```

### 2. 创建虚拟环境（推荐）

```bash
cd GPU_CollisionDetecction
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：根据您的CUDA版本选择正确的CuPy包：
- CUDA 11.x: `pip install cupy-cuda11x`
- CUDA 12.x: `pip install cupy-cuda12x`

### 4. 验证安装

```bash
python -c "import cupy as cp; print('CuPy version:', cp.__version__); print('CUDA available:', cp.cuda.is_available())"
```

## 项目结构

```
GPU_CollisionDetecction/
├── src/                    # 核心源代码
│   ├── __init__.py
│   ├── rigid_body.py      # 刚体物理系统
│   ├── spatial_grid.py    # 空间网格数据结构
│   ├── kernels.py         # CUDA内核函数
│   ├── simulator.py       # 主仿真器
│   ├── visualizer.py      # 可视化工具
│   └── performance.py     # 性能监控
├── examples/              # 示例程序
│   └── gravity_fall.py    # 重力下落场景
├── tests/                 # 测试脚本
│   └── benchmark.py       # 性能基准测试
├── output/                # 输出文件（视频、图表等）
├── algorithm_design.md    # 算法设计文档
├── cuda_cupy_implementation.md  # CuPy实现方案
├── requirements.txt       # Python依赖
└── README.md             # 本文件
```

## 快速开始

### 示例1：重力下落场景

模拟8000个小球在重力作用下下落并碰撞：

```bash
python examples/gravity_fall.py
```

**输出**：
- 实时3D可视化窗口
- `output/gravity_fall.mp4` - 仿真动画视频
- `output/gravity_fall_performance.png` - 性能分析图
- `output/gravity_fall_metrics.csv` - 详细性能数据

**参数调整**（在脚本中修改）：
```python
NUM_OBJECTS = 8000          # 物体数量（建议5000-15000）
WORLD_BOUNDS = ((-25, 0, -25), (25, 50, 25))
CELL_SIZE = 2.5             # 网格单元大小
NUM_FRAMES = 600            # 总帧数
```

### 性能测试

运行基准测试以找到最佳配置：

```bash
python tests/benchmark.py
```

**输出**：
- `output/benchmark_scaling.png` - 性能扩展曲线
- `output/benchmark_results.csv` - 详细测试结果

测试将评估不同物体数量（1000-15000）下的性能表现。

## 核心算法

### 1. 空间分割（Uniform Grid）

- 将3D空间划分为规则网格
- 使用空间哈希加速查找
- 时间复杂度：O(N)

### 2. 碰撞检测流程

```
1. Grid Build: 构建空间网格
   ├─ 计算每个物体的网格哈希
   ├─ 按哈希值排序
   └─ 找到每个单元的起始/结束位置

2. Broad Phase: 粗检测
   └─ 检查每个物体周围27个单元（3x3x3）

3. Narrow Phase: 精确检测
   └─ 球体-球体相交测试

4. Collision Response: 碰撞响应
   ├─ 冲量法计算速度变化
   └─ 位置修正避免穿透

5. Physics Integration: 物理积分
   ├─ 半隐式Euler方法
   └─ 边界碰撞处理
```

### 3. GPU并行化

所有计算密集型操作都在GPU上并行执行：

- 网格哈希计算：每个线程处理一个物体
- 碰撞检测：每个线程检查一个物体的邻居
- 碰撞响应：每个线程处理一个碰撞对
- 物理积分：每个线程更新一个物体

## 性能特性

### RTX 3050 性能（参考）

| 物体数量 | 平均帧时间 | FPS | 平均碰撞数 |
|---------|-----------|-----|-----------|
| 1,000   | ~2ms      | 500+ | ~100     |
| 5,000   | ~8ms      | 125  | ~800     |
| 8,000   | ~15ms     | 66   | ~1500    |
| 10,000  | ~20ms     | 50   | ~2200    |
| 15,000  | ~35ms     | 28   | ~4500    |

**建议配置**：
- 实时可视化：8,000物体
- 离线渲染：15,000物体
- 最佳性能：5,000-8,000物体

### 性能优化技巧

1. **调整网格大小**
   - `cell_size = 2.0 * average_radius`
   - 太小：单元过多，占用显存
   - 太大：每个单元物体过多，检测慢

2. **减少可视化频率**
   ```python
   FRAME_SKIP = 2  # 每2帧显示一次
   ```

3. **控制物体数量**
   - RTX 3050: 5000-10000物体最优
   - 更高端显卡可增加

## API使用示例

### 基础使用

```python
from src import PhysicsSimulator
import numpy as np

# 创建仿真器
sim = PhysicsSimulator(
    num_objects=5000,
    world_bounds=((-50, -50, -50), (50, 50, 50)),
    cell_size=2.0,
    gravity=(0, -9.8, 0)
)

# 设置初始状态
positions = np.random.uniform(-40, 40, (5000, 3))
sim.bodies.set_positions(positions)

velocities = np.random.uniform(-5, 5, (5000, 3))
sim.bodies.set_velocities(velocities)

# 仿真循环
for frame in range(1000):
    step_info = sim.step()
    print(f"Frame {frame}: {step_info['num_collisions']} collisions")
    
    # 获取结果（每N帧）
    if frame % 10 == 0:
        data = sim.bodies.to_cpu()
        # 使用data['positions'], data['velocities']等
```

### 性能监控

```python
from src import PerformanceMonitor

monitor = PerformanceMonitor()

for frame in range(1000):
    step_info = sim.step()
    monitor.record_metric('frame_time', step_info['total_time'])

# 输出统计
monitor.print_statistics()

# 绘制图表
monitor.plot_timeline(save_path='performance.png')
```

## 技术细节

### 数据布局

使用**SOA（Structure of Arrays）**布局优化GPU内存访问：

```python
# SOA布局（推荐）
positions = [[x1, y1, z1], [x2, y2, z2], ...]  # [N, 3]
velocities = [[vx1, vy1, vz1], [vx2, vy2, vz2], ...]

# 优点：连续内存访问，高缓存命中率
```

### CUDA内核优化

- 线程块大小：256（针对RTX 3050优化）
- 合并内存访问
- 原子操作最小化
- 共享内存使用（计划中）

## 故障排除

### 1. CUDA内存不足

**错误**：`CUDARuntimeError: out of memory`

**解决**：
- 减少`NUM_OBJECTS`
- 增加`CELL_SIZE`（减少网格单元数）
- 减少`max_pairs`（在simulator.py中）

### 2. 性能过低

**原因**：
- 物体过于密集
- 网格大小不合适
- 显存带宽瓶颈

**解决**：
- 调整`CELL_SIZE`
- 增加物体间距
- 减少可视化频率

### 3. CuPy安装失败

**问题**：CUDA版本不匹配

**解决**：
```bash
# 检查CUDA版本
nvcc --version

# 安装对应版本
pip install cupy-cuda11x  # 或 cupy-cuda12x
```

## 未来改进

- [ ] 支持更多形状（盒子、胶囊体等）
- [ ] 实现BVH（层次包围盒）加速结构
- [ ] 3D Gaussian Splatting场景集成
- [ ] 多GPU支持
- [ ] 实时交互控制
- [ ] 软体物理

## 许可证

MIT License

## 作者

Cindy - 计算机动画课程项目 2025-26

## 参考文献

1. "Real-Time Collision Detection" - Christer Ericson
2. "GPU Gems 3: Chapter 32. Broad-Phase Collision Detection with CUDA"
3. "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
4. CuPy Documentation: https://docs.cupy.dev/

## 致谢

感谢CuPy项目提供的优秀GPU加速框架。
