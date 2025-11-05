# 快速开始指南

## 第一步：安装依赖

### 1. 检查CUDA环境

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version
```

### 2. 创建虚拟环境

```bash
cd GPU_CollisionDetecction

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖包

```bash
# 根据您的CUDA版本安装CuPy
# CUDA 11.x:
pip install cupy-cuda11x

# CUDA 12.x:
pip install cupy-cuda12x

# 安装其他依赖
pip install numpy scipy matplotlib opencv-python
```

## 第二步：验证安装

```bash
# 运行简单测试
python tests/simple_test.py
```

如果看到"All tests passed!"，说明安装成功！

## 第三步：运行示例

### 示例1：重力下落场景（推荐）

```bash
python examples/gravity_fall.py
```

这将：
1. 模拟8000个小球下落并碰撞
2. 显示实时3D可视化
3. 生成视频文件：`output/gravity_fall.mp4`
4. 生成性能分析图：`output/gravity_fall_performance.png`

**预计运行时间**：约5-10分钟（取决于硬件）

### 自定义参数

编辑 `examples/gravity_fall.py`，修改以下参数：

```python
# 基础配置
NUM_OBJECTS = 8000          # 物体数量（建议5000-15000）
CELL_SIZE = 2.5             # 网格大小
NUM_FRAMES = 600            # 总帧数（600帧 = 10秒）

# 可选配置
SAVE_VIDEO = True           # 是否保存视频
SHOW_REALTIME = True        # 是否显示实时可视化
FRAME_SKIP = 2              # 可视化跳帧（提高性能）
```

## 第四步：性能测试

```bash
# 运行基准测试
python tests/benchmark.py
```

这将测试不同规模（1000-15000物体）下的性能，生成：
- `output/benchmark_scaling.png` - 性能曲线图
- `output/benchmark_results.csv` - 详细数据

**预计运行时间**：约10-20分钟

## 常见问题

### Q1: 显存不足错误

**错误信息**：`CUDARuntimeError: out of memory`

**解决方法**：
1. 减少物体数量：`NUM_OBJECTS = 5000`
2. 增加网格大小：`CELL_SIZE = 3.0`
3. 关闭实时可视化：`SHOW_REALTIME = False`

### Q2: 性能过低（FPS < 30）

**解决方法**：
1. 减少物体数量
2. 增加可视化跳帧：`FRAME_SKIP = 4`
3. 关闭视频保存：`SAVE_VIDEO = False`
4. 确保没有其他程序占用GPU

### Q3: CuPy导入失败

**错误信息**：`Import "cupy" could not be resolved`

**解决方法**：
1. 确认CUDA已正确安装
2. 根据CUDA版本安装正确的CuPy：
   ```bash
   # 查看CUDA版本
   nvcc --version
   
   # 安装对应版本
   pip install cupy-cuda11x  # 或 cupy-cuda12x
   ```

### Q4: 可视化窗口卡顿

**解决方法**：
1. 增加跳帧：`FRAME_SKIP = 3`
2. 减少物体数量
3. 使用离线渲染（只保存视频，不显示实时可视化）

## RTX 3050 推荐配置

基于测试，以下是RTX 3050的推荐配置：

### 实时可视化模式
```python
NUM_OBJECTS = 8000
CELL_SIZE = 2.5
FRAME_SKIP = 2
SAVE_VIDEO = True
SHOW_REALTIME = True
```
预期性能：~60 FPS

### 高质量离线渲染
```python
NUM_OBJECTS = 12000
CELL_SIZE = 2.5
FRAME_SKIP = 1
SAVE_VIDEO = True
SHOW_REALTIME = False  # 关闭实时显示
```
预期性能：~40 FPS（但不影响体验）

### 快速测试模式
```python
NUM_OBJECTS = 3000
CELL_SIZE = 2.0
NUM_FRAMES = 300  # 5秒
SAVE_VIDEO = False
SHOW_REALTIME = True
```
预期性能：100+ FPS

## 下一步

1. **查看算法文档**：
   - `algorithm_design.md` - 详细的算法设计
   - `cuda_cupy_implementation.md` - CuPy实现方案

2. **修改和实验**：
   - 尝试不同的初始条件
   - 调整物理参数（重力、弹性系数等）
   - 创建新的场景

3. **性能优化**：
   - 运行benchmark找到最佳配置
   - 分析性能瓶颈
   - 尝试不同的网格大小

4. **扩展功能**：
   - 添加新的物体形状
   - 实现用户交互
   - 集成3D Gaussian Splatting场景

## 获取帮助

如果遇到问题：
1. 查看 `README.md` 的"故障排除"部分
2. 检查终端输出的错误信息
3. 运行 `tests/simple_test.py` 验证系统状态

祝使用愉快！
