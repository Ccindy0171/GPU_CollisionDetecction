# GPU 碰撞检测系统 - 中文文档

## 目录

1. [程序运行环境](#程序运行环境)
2. [程序模块逻辑关系](#程序模块逻辑关系)
3. [程序运行流程](#程序运行流程)
4. [功能演示方法](#功能演示方法)
5. [参考文献](#参考文献)

---

## 程序运行环境

### 硬件要求

- **GPU**: NVIDIA GPU，支持 CUDA（计算能力 3.5+）
- **测试平台**: RTX 3050 (4GB 显存)、RTX 3060 (12GB)、GTX 1660 Ti (6GB)
- **最低显存**: 2GB（用于 500-1000 个物体）
- **推荐显存**: 4GB+（用于 2000+ 个物体）
- **系统内存**: 最低 8GB，推荐 16GB

### 软件环境

- **Python 版本**: 3.8 - 3.11（推荐 3.10）
- **CUDA 版本**: 11.x 或 12.x（推荐 12.x）
- **操作系统**:
  - ✅ Linux（Ubuntu 20.04+ 及其他发行版）
  - ✅ Windows 10/11（支持 WSL2）
  - ⚠️  macOS（需要外置 NVIDIA GPU 支持 CUDA）

### 依赖库

#### 核心依赖

```
Python >= 3.8
CUDA >= 11.0
```

#### Python 包依赖

```
cupy-cuda12x >= 13.6.0      # GPU 计算（CUDA 12.x）
numpy >= 1.21.0              # 数值计算
scipy >= 1.7.0               # 科学计算
PyOpenGL >= 3.1.10           # 3D 可视化
opencv-python >= 4.5.0       # 视频录制
```

### 安装步骤

1. **安装 CUDA 工具包**
   ```bash
   # 验证 CUDA 安装
   nvcc --version
   ```

2. **创建 Python 虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**
   ```bash
   python check_system.py
   ```

---

## 程序模块逻辑关系

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       应用层                                  │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ examples/        │  │ tests/           │                │
│  │ 示例程序         │  │ 测试程序         │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 调用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      核心库 (src/)                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         PhysicsSimulator (simulator.py)              │  │
│  │         主物理仿真器 - 协调所有组件                    │  │
│  └──────────────────────────────────────────────────────┘  │
│           │           │            │           │            │
│           ▼           ▼            ▼           ▼            │
│  ┌────────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐     │
│  │ RigidBody  │ │ Spatial │ │  CUDA   │ │Performance│     │
│  │ 刚体系统   │ │ 空间网格 │ │  核函数  │ │ 性能监控  │     │
│  │            │ │         │ │         │ │           │     │
│  │• 位置      │ │• 均匀网格│ │• 哈希   │ │• 计时     │     │
│  │• 速度      │ │• 空间哈希│ │• 检测   │ │• 统计     │     │
│  │• 力        │ │         │ │• 响应   │ │           │     │
│  └────────────┘ └─────────┘ └─────────┘ └──────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      OpenGLVisualizer (opengl_visualizer.py)         │  │
│  │      可视化模块 - OpenGL 渲染和视频录制               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 依赖
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    外部依赖库                                 │
│  • CuPy (GPU 计算)                                          │
│  • NumPy (数值数组)                                         │
│  • PyOpenGL (3D 渲染)                                       │
│  • OpenCV (视频处理)                                        │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块说明

#### 1. PhysicsSimulator (simulator.py)
**功能**: 主物理仿真器，协调所有组件
- 初始化物理系统
- 执行仿真循环
- 协调碰撞检测流程
- 管理性能指标

#### 2. RigidBodySystem (rigid_body.py)
**功能**: 管理所有物理对象及其属性
- 存储对象位置、速度、受力
- 存储物理属性（质量、半径、弹性系数）
- 提供 CPU↔GPU 数据传输方法
- 计算系统动能、动量

#### 3. UniformGrid (spatial_grid.py)
**功能**: 空间加速结构，用于快速邻近查询
- 将 3D 空间划分为规则网格单元
- 将位置映射到网格单元（空间哈希）
- 存储对象到单元的映射关系
- 提供邻近单元查询

#### 4. CUDA Kernels (kernels.py)
**功能**: GPU 加速的计算核函数
- `COMPUTE_GRID_HASH_KERNEL`: 位置 → 网格哈希映射
- `FIND_CELL_START_KERNEL`: 构建单元起始/结束索引
- `BROAD_PHASE_KERNEL`: 检测碰撞对
- `COLLISION_RESPONSE_KERNEL`: 解决碰撞（冲量法）
- `INTEGRATE_KERNEL`: 物理积分（半隐式 Euler 方法）

#### 5. OpenGLVisualizer (opengl_visualizer.py)
**功能**: 3D 可视化和渲染
- Phong 光照模型
- 相机控制（旋转、缩放、平移）
- 实时渲染
- 视频录制（MP4 导出）

#### 6. PerformanceMonitor (performance.py)
**功能**: 性能跟踪和分析
- GPU/CPU 事件计时
- 指标收集和统计
- 性能可视化（图表）
- CSV 数据导出

### 数据流向

#### 初始化阶段

```
用户脚本
    │
    ├─> 创建 PhysicsSimulator
    │       │
    │       ├─> 初始化 RigidBodySystem (在 GPU 上)
    │       ├─> 初始化 UniformGrid (在 GPU 上)
    │       └─> 编译 CUDA 核函数
    │
    ├─> 设置初始位置 (CPU → GPU)
    ├─> 设置速度、质量、半径 (CPU → GPU)
    └─> (可选) 创建 OpenGLVisualizer
```

#### 仿真循环（每帧）

```
PhysicsSimulator.step()
    │
    ├─> 1. 积分（更新位置/速度）
    │   │   • 应用力（重力、外力）
    │   │   • 半隐式 Euler 积分
    │   │   • 处理边界碰撞
    │   │   [CUDA 核函数: INTEGRATE_KERNEL]
    │   └─> 更新的位置、速度 (在 GPU 上)
    │
    ├─> 2. 构建网格（空间分区）
    │   │   • 为每个对象计算网格哈希
    │   │   • 按哈希值排序对象
    │   │   • 找到单元起始/结束索引
    │   │   [CUDA 核函数: COMPUTE_GRID_HASH, FIND_CELL_START]
    │   └─> 网格数据结构 (在 GPU 上)
    │
    ├─> 3. 检测碰撞（粗检测阶段）
    │   │   • 检查每个对象的 27 个邻近单元
    │   │   • 球-球相交测试
    │   │   • 构建碰撞对列表
    │   │   [CUDA 核函数: BROAD_PHASE_KERNEL]
    │   └─> 碰撞对数组 (在 GPU 上)
    │
    ├─> 4. 解决碰撞（碰撞响应）
    │   │   • 计算碰撞法线
    │   │   • 应用冲量（速度变化）
    │   │   • 位置校正（分离）
    │   │   [CUDA 核函数: COLLISION_RESPONSE_KERNEL]
    │   └─> 更新的速度、位置 (在 GPU 上)
    │
    ├─> 5. (可选) 可视化
    │   │   • 将位置、颜色复制到 CPU
    │   │   • 使用 OpenGL 渲染
    │   └─> 显示帧 / 保存到视频
    │
    └─> 返回统计信息
```

---

## 程序运行流程

### 主要算法流程

#### 1. 物理积分步骤

```
对于每个对象 (在 GPU 上并行):
    1. 读取: 位置、速度、受力、质量、半径
    2. 计算: 加速度 = 受力/质量 + 重力
    3. 更新: 速度 += 加速度 × dt  (半隐式)
    4. 更新: 位置 += 速度 × dt
    5. 应用: 速度阻尼（空气阻力）
    6. 检查: 边界碰撞
        如果碰到边界:
            - 将位置钳制到边界
            - 反转速度分量
            - 应用弹性系数
    7. 写入: 更新的位置、速度
```

#### 2. 网格构建步骤

```
阶段 A - 计算哈希:
    对于每个对象 (并行):
        1. grid_coord = floor((位置 - 世界最小值) / 单元大小)
        2. hash = z × (res_y × res_x) + y × res_x + x
        3. 存储哈希值

阶段 B - 排序:
    按哈希值对对象索引排序 (GPU 并行排序)

阶段 C - 找到单元边界:
    对于每个对象 (并行):
        如果是单元中的第一个: 标记 cell_start[hash] = index
        如果是单元中的最后一个: 标记 cell_end[hash] = index + 1
```

#### 3. 碰撞检测步骤（粗检测）

```
对于每个对象 i (并行):
    1. 确定对象 i 的网格单元
    2. 对于 27 个邻近单元中的每一个:
        3. 获取该单元中的对象（通过 cell_start/end）
        4. 对于单元中的每个对象 j:
            如果 i < j:  # 避免重复
                5. dist = ||pos_i - pos_j||
                6. 如果 dist < radius_i + radius_j:
                    7. 原子地将对 (i,j) 添加到碰撞列表
```

#### 4. 碰撞响应步骤

```
对于每个碰撞对 (i,j) (并行):
    1. 计算碰撞法线: n = (pos_j - pos_i) / ||pos_j - pos_i||
    2. 计算相对速度: v_rel = vel_j - vel_i
    3. 计算沿法线的速度: v_n = v_rel · n
    
    如果 v_n < 0 (接近):
        4. 计算冲量: j = -(1+e) × v_n / (1/m_i + 1/m_j)
        5. 应用冲量到速度:
            vel_i -= j × n / m_i  (原子操作)
            vel_j += j × n / m_j  (原子操作)
    
    如果穿透:
        6. 计算穿透深度: p = (r_i + r_j) - dist
        7. 按质量比例分离对象:
            pos_i -= correction_i × n  (原子操作)
            pos_j += correction_j × n  (原子操作)
```

### 算法复杂度分析

- **无加速结构**: O(N²) - 检查所有对
- **使用均匀网格**: O(N×k)，其中 k = 每个单元的平均对象数
- **最佳情况**: O(N)，当对象均匀分布时

**单元大小选择**:
- 最优: 2× 平均对象直径
- 太小: 过多内存，需要检查更多单元
- 太大: 每个单元对象太多，碰撞检查变慢

---

## 功能演示方法

### 快速开始

#### 1. 系统验证

运行系统检查工具验证环境:

```bash
python check_system.py
```

预期输出:
```
✓ Python 3.10 兼容
✓ 找到 CUDA 12.2
✓ RTX 3050: 4.00GB 显存
✓ CuPy 13.6.0 可用
✓ GPU 加速: 47.2×
```

#### 2. 主演示程序 - 重力下落

模拟球体在重力作用下下落、碰撞并堆积:

```bash
python examples/gravity_fall.py
```

**功能特点**:
- 实时 3D 可视化
- 交互式相机控制
- 高质量视频录制
- 性能统计显示

**交互控制**:
| 键/鼠标 | 功能 |
|---------|------|
| 左键拖动 | 旋转相机 |
| 右键拖动 | 缩放 |
| 中键拖动 | 平移相机 |
| 空格 | 暂停/继续 |
| G | 显示/隐藏网格 |
| R | 重置相机 |
| Q/ESC | 退出 |

**配置参数** (在脚本中编辑):
```python
NUM_OBJECTS = 500           # 球体数量
WORLD_BOUNDS = ((-20, 0, -20), (20, 40, 20))  # 世界边界
CELL_SIZE = 2.0             # 网格单元大小
NUM_FRAMES = 600            # 帧数（10秒 @ 60fps）
RECORD_VIDEO = True         # 是否录制视频
```

#### 3. 单元测试

**测试 1: 正面碰撞**
```bash
python tests/test_01_head_on.py
```
验证两个球体正面碰撞后的速度反转和能量守恒。

**测试 2: 重叠分离**
```bash
python tests/test_02_static_overlap.py
```
测试初始重叠球体的分离算法。

**测试 3: 多球下落**
```bash
python tests/test_03_falling_balls.py
```
测试多体交互和重力作用。

**测试 4: 大规模测试**
```bash
python tests/test_04_large_scale.py --objects 1000
```
压力测试，可配置对象数量。

#### 4. 物理仿真（无可视化）

仅运行物理计算，不显示图形界面:

```bash
python tests/test_physics_only.py
```

适用于:
- 无显示服务器环境
- 性能基准测试
- 批处理模拟

#### 5. 性能基准测试

运行完整的性能分析:

```bash
python tests/benchmark.py
```

生成:
- 性能曲线图
- 统计数据 CSV
- 详细报告

### 配置示例

#### 基本仿真

```python
from src.simulator import PhysicsSimulator
import numpy as np

# 创建仿真器
sim = PhysicsSimulator(
    num_objects=1000,
    world_bounds=((-50, 0, -50), (50, 50, 50)),
    cell_size=2.0,
    dt=1.0/60.0,
    gravity=(0, -9.81, 0)
)

# 设置初始状态
positions = np.random.uniform(-40, 40, (1000, 3)).astype(np.float32)
sim.bodies.set_positions(positions)

# 运行仿真
for frame in range(600):
    stats = sim.step()
    print(f"帧 {frame}: {stats['num_collisions']} 次碰撞")
```

#### 自定义场景

```python
# 爆炸效果
center = np.array([0, 25, 0])
positions = sim.bodies.positions.get()
directions = positions - center
distances = np.linalg.norm(directions, axis=1, keepdims=True)
velocities = (directions / distances) * 20.0  # 20 m/s 向外
sim.bodies.set_velocities(velocities)
```

### 常见问题解决

#### 问题 1: "CUDA_ERROR_OUT_OF_MEMORY"

**原因**: GPU 显存不足

**解决方案**:
```python
# 减少对象数量
NUM_OBJECTS = 500

# 增大单元大小
CELL_SIZE = 3.0

# 关闭可视化
RECORD_VIDEO = False
```

#### 问题 2: 低帧率

**解决方案**:
- 减少对象数量
- 增大单元大小（减少网格分辨率）
- 降低渲染分辨率
- 使用无头模式

#### 问题 3: 对象穿透地板

**原因**: 时间步长太大

**解决方案**:
```python
# 减小时间步长
dt = 1.0/120.0  # 而不是 1/60

# 或减小重力
gravity = (0, -5.0, 0)  # 而不是 -9.81
```

---

## 参考文献

### 研究论文与书籍

1. **Ericson, C.** "Real-Time Collision Detection" - CRC Press, 2004
   - 碰撞检测算法的综合参考

2. **Green, S.** "GPU Gems 3: Chapter 32 - Broad-Phase Collision Detection with CUDA" - NVIDIA, 2007
   - GPU 加速碰撞检测技术

3. **Millington, I.** "Game Physics Engine Development" - CRC Press, 2010
   - 物理仿真和碰撞响应方法

4. **Witkin, A. & Baraff, D.** "Physically Based Modeling: Principles and Practice" - SIGGRAPH Course Notes
   - 数值积分和物理基础

### 算法与技术

5. **空间哈希**: Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects" (2003)

6. **冲量响应**: Baraff, "Fast Contact Force Computation for Nonpenetrating Rigid Bodies" (1994)

7. **半隐式 Euler**: Verlet Integration and Symplectic Methods in Physics

### 开源库与框架

8. **CuPy**: GPU 加速数组库
   - 文档: https://docs.cupy.dev/
   - GitHub: https://github.com/cupy/cupy

9. **PyOpenGL**: OpenGL 的 Python 绑定
   - 文档: http://pyopengl.sourceforge.net/

10. **NumPy**: 科学计算基础包
    - 文档: https://numpy.org/doc/

### CUDA 与 GPU 计算

11. **NVIDIA CUDA 文档**
    - 编程指南: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    - 最佳实践: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

### 代码引用与致谢

- **空间网格结构**: 参考 NVIDIA PhysX 和 Bullet Physics 实现
- **CUDA 核函数优化**: 基于 NVIDIA CUDA 示例和最佳实践
- **OpenGL 可视化**: 使用标准 Phong 光照模型和 GLU 基本体
- **视频编码**: 通过 OpenCV 使用 FFmpeg/H.264 编码

---

本文档版本: 1.0  
更新日期: 2025 年 1 月
