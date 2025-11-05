# 🎯 GPU碰撞检测系统 - 使用指南

## 📋 项目概述

这是一个**完整的、生产级的GPU加速碰撞检测与物理仿真系统**，基于CuPy实现。

### ✨ 核心功能

1. ✅ **高效最近邻查找** - 基于Uniform Grid的O(N)算法
2. ✅ **完整物理仿真** - 刚体动力学、碰撞响应、边界处理
3. ✅ **性能测试分析** - 详细的基准测试和性能监控
4. ✅ **专业可视化** - 实时3D显示和高质量视频导出

### 🎬 效果展示

- 支持**8000+物体**实时仿真（RTX 3050）
- 达到**60+ FPS**性能
- 准确的物理碰撞效果
- 高质量MP4视频输出

---

## 🚀 快速开始（3步搞定）

### 第1步：安装依赖

```bash
cd GPU_CollisionDetecction

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖（根据您的CUDA版本选择）
pip install cupy-cuda11x  # CUDA 11.x
# 或
pip install cupy-cuda12x  # CUDA 12.x

# 安装其他依赖
pip install numpy scipy matplotlib opencv-python
```

### 第2步：验证安装

```bash
python tests/simple_test.py
```

看到 "✓ All tests passed!" 即可继续。

### 第3步：运行示例

```bash
# 使用启动脚本（推荐）
./run.sh

# 或直接运行示例
python examples/gravity_fall.py
```

**就这么简单！** 🎉

---

## 📖 详细使用说明

### 示例1：重力下落场景

这是**主要演示示例**，展示了完整功能。

#### 运行方式：

```bash
python examples/gravity_fall.py
```

#### 功能说明：

- 模拟8000个小球在重力作用下下落
- 小球具有不同的半径、质量和弹性
- 实时显示3D动画
- 自动保存视频和性能数据

#### 输出文件：

```
output/
├── gravity_fall.mp4                    # 仿真动画（~100MB）
├── gravity_fall_performance.png       # 性能曲线图
└── gravity_fall_metrics.csv           # 详细性能数据
```

#### 预计耗时：
- 计算+渲染：约5-10分钟
- 仅计算（关闭可视化）：约2-3分钟

#### 自定义参数：

编辑 `examples/gravity_fall.py`，修改顶部配置：

```python
# === 可调整参数 ===
NUM_OBJECTS = 8000          # 物体数量（推荐：5000-15000）
WORLD_BOUNDS = ((-25, 0, -25), (25, 50, 25))
CELL_SIZE = 2.5             # 网格大小
NUM_FRAMES = 600            # 总帧数（600=10秒@60fps）

# 性能优化选项
SAVE_VIDEO = True           # 是否保存视频
SHOW_REALTIME = True        # 是否显示实时可视化
FRAME_SKIP = 2              # 可视化跳帧（2=每2帧显示1次）
```

---

### 示例2：性能基准测试

测试不同规模下的性能，找到最佳配置。

#### 运行方式：

```bash
python tests/benchmark.py
```

#### 测试内容：

测试物体数量：1000, 2000, 5000, 8000, 10000, 15000

每个规模运行100帧，统计：
- 平均帧时间
- FPS性能
- 碰撞数量
- 网格占用率

#### 输出文件：

```
output/
├── benchmark_scaling.png              # 性能扩展曲线
└── benchmark_results.csv              # 详细测试数据
```

#### 预计耗时：
- 完整测试：约10-20分钟

#### 测试结果示例（RTX 3050）：

```
物体数量 | 平均帧时间 | FPS  | 平均碰撞数
---------|-----------|------|----------
1,000    | ~2ms      | 500+ | ~100
5,000    | ~8ms      | 125  | ~800
8,000    | ~15ms     | 66   | ~1500
10,000   | ~20ms     | 50   | ~2200
15,000   | ~35ms     | 28   | ~4500
```

---

## ⚙️ 配置建议

### RTX 3050 推荐配置

#### 🎯 平衡模式（推荐）
```python
NUM_OBJECTS = 8000
CELL_SIZE = 2.5
FRAME_SKIP = 2
```
- 性能：~60 FPS
- 视觉效果：流畅
- 适合：实时演示

#### 🚀 高性能模式
```python
NUM_OBJECTS = 5000
CELL_SIZE = 2.0
FRAME_SKIP = 1
```
- 性能：100+ FPS
- 视觉效果：非常流畅
- 适合：性能测试

#### 🎬 高质量模式
```python
NUM_OBJECTS = 12000
CELL_SIZE = 2.5
SHOW_REALTIME = False  # 关闭实时显示
SAVE_VIDEO = True
```
- 性能：~40 FPS（不影响体验）
- 视觉效果：更多物体
- 适合：离线渲染

#### ⚡ 快速测试模式
```python
NUM_OBJECTS = 3000
NUM_FRAMES = 180  # 3秒
SAVE_VIDEO = False
```
- 性能：150+ FPS
- 适合：快速验证

---

## 🔧 故障排除

### ❌ 问题1：显存不足

**症状**：
```
CUDARuntimeError: out of memory
```

**解决方案**：
1. 减少物体数量：`NUM_OBJECTS = 5000`
2. 增大网格：`CELL_SIZE = 3.0`
3. 关闭可视化：`SHOW_REALTIME = False`

### ❌ 问题2：性能过低

**症状**：FPS < 30

**解决方案**：
1. 减少物体：`NUM_OBJECTS = 6000`
2. 增加跳帧：`FRAME_SKIP = 4`
3. 确保GPU空闲（关闭其他程序）

### ❌ 问题3：CuPy导入失败

**症状**：
```
ModuleNotFoundError: No module named 'cupy'
```

**解决方案**：
```bash
# 检查CUDA版本
nvcc --version

# 安装对应版本
pip install cupy-cuda11x  # 或 cupy-cuda12x
```

### ❌ 问题4：可视化窗口无响应

**症状**：窗口卡住不更新

**解决方案**：
1. 增加跳帧：`FRAME_SKIP = 3`
2. 使用离线模式：`SHOW_REALTIME = False`

---

## 📚 文档索引

### 📖 用户文档
- **README.md** - 完整项目说明
- **QUICKSTART.md** - 快速开始指南
- **本文档** - 详细使用说明

### 🔬 技术文档
- **algorithm_design.md** - 算法设计详解
- **cuda_cupy_implementation.md** - CUDA实现方案
- **PROJECT_SUMMARY.md** - 项目总结

### 💻 代码文档
- 每个模块都有详细的docstring
- 关键算法有行内注释
- CUDA kernel有详细说明

---

## 🎓 学习路径

### 初学者路径

1. **运行示例**
   ```bash
   python examples/gravity_fall.py
   ```

2. **查看输出**
   - 观察实时可视化
   - 查看生成的视频
   - 分析性能数据

3. **调整参数**
   - 修改物体数量
   - 改变重力大小
   - 尝试不同弹性系数

### 进阶路径

1. **性能测试**
   ```bash
   python tests/benchmark.py
   ```

2. **阅读代码**
   - `src/simulator.py` - 主仿真流程
   - `src/kernels.py` - GPU算法实现
   - `src/rigid_body.py` - 物理系统

3. **理解算法**
   - 阅读 `algorithm_design.md`
   - 研究空间分割原理
   - 分析碰撞检测流程

### 高级路径

1. **修改算法**
   - 实现新的碰撞形状
   - 优化网格构建
   - 添加新的物理效果

2. **性能优化**
   - 调整线程块大小
   - 优化内存访问模式
   - 减少原子操作

3. **功能扩展**
   - 添加用户交互
   - 实现软体物理
   - 集成渲染引擎

---

## 📊 API参考

### PhysicsSimulator

主仿真器类。

```python
from src import PhysicsSimulator

sim = PhysicsSimulator(
    num_objects=5000,                              # 物体数量
    world_bounds=((-50,-50,-50), (50,50,50)),    # 世界边界
    cell_size=2.0,                                # 网格大小
    dt=1.0/60.0,                                  # 时间步长
    gravity=(0.0, -9.8, 0.0),                     # 重力
    damping=0.01                                  # 阻尼
)

# 设置物体属性
sim.bodies.set_positions(positions)      # [N, 3]
sim.bodies.set_velocities(velocities)    # [N, 3]
sim.bodies.set_radii(radii)             # [N]
sim.bodies.set_masses(masses)           # [N]

# 仿真循环
for frame in range(1000):
    step_info = sim.step()  # 执行一帧
    
    # step_info包含：
    # - num_collisions: 碰撞数量
    # - total_time: 总耗时(ms)
    # - grid_build_time: 网格构建时间
    # - collision_detect_time: 碰撞检测时间
    # - collision_resolve_time: 碰撞响应时间
    # - integrate_time: 物理积分时间
```

### RealtimeVisualizer

实时3D可视化。

```python
from src import RealtimeVisualizer

viz = RealtimeVisualizer(
    world_bounds=((-50,-50,-50), (50,50,50)),
    figsize=(12, 10),
    title="My Simulation"
)

# 更新显示
data = sim.bodies.to_cpu()
viz.update(
    positions=data['positions'],
    colors=data['colors'],
    radii=data['radii'],
    info_text="Frame: 100"
)

# 保存图片
viz.save_frame('frame.png')

# 关闭
viz.close()
```

### PerformanceMonitor

性能监控。

```python
from src import PerformanceMonitor

monitor = PerformanceMonitor()

# 记录指标
for frame in range(1000):
    step_info = sim.step()
    monitor.record_metric('frame_time', step_info['total_time'])

# 输出统计
monitor.print_statistics()

# 绘制图表
monitor.plot_timeline(save_path='performance.png')

# 导出数据
monitor.export_to_csv('metrics.csv')
```

---

## 🤝 常见问题（FAQ）

### Q1: 需要什么硬件？
**A**: NVIDIA GPU（支持CUDA），推荐4GB+显存。项目在RTX 3050上测试。

### Q2: 支持AMD显卡吗？
**A**: 不支持。项目使用CUDA，只能在NVIDIA显卡上运行。

### Q3: 可以在CPU上运行吗？
**A**: 理论上可以（将CuPy改为NumPy），但性能会慢100倍以上。

### Q4: 如何提高性能？
**A**: 
1. 减少物体数量
2. 增大网格单元
3. 减少可视化频率
4. 使用更好的GPU

### Q5: 能处理多少物体？
**A**: 取决于显存大小：
- 4GB显存：10000-15000物体
- 8GB显存：30000+物体
- 12GB+显存：50000+物体

### Q6: 如何导出数据？
**A**: 
```python
data = sim.bodies.to_cpu()
np.save('positions.npy', data['positions'])
```

### Q7: 可以实时交互吗？
**A**: 当前版本不支持，但可以扩展添加。

### Q8: 支持哪些操作系统？
**A**: Linux和Windows（需要CUDA支持）。

---

## 📞 获取帮助

如果遇到问题：

1. **查看文档**
   - README.md 的"故障排除"
   - 本文档的"故障排除"部分

2. **运行测试**
   ```bash
   python tests/simple_test.py
   ```

3. **检查输出**
   - 查看终端错误信息
   - 检查GPU使用情况（`nvidia-smi`）

4. **降低复杂度**
   - 减少物体数量
   - 关闭可视化
   - 缩短仿真时间

---

## 🎉 开始使用

一切准备就绪！运行以下命令开始：

```bash
# 快速启动
./run.sh

# 或直接运行示例
python examples/gravity_fall.py
```

**祝您使用愉快！** 🚀

---

## 📝 更新日志

### v1.0.0 (2025-11-04)
- ✅ 完整实现所有核心功能
- ✅ 详细文档和注释
- ✅ 性能测试和优化
- ✅ 示例程序和教程

---

**项目作者**: Cindy  
**课程**: 计算机动画 2025-26  
**日期**: 2025-11-04
