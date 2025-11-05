# 基于CUDA (CuPy)的碰撞检测算法实现方案

## 1. CuPy简介与优势

### 1.1 为什么选择CuPy？

**CuPy**是一个GPU加速的NumPy兼容库，提供了：
- **NumPy兼容API**：几乎零学习成本
- **自定义CUDA Kernel**：通过RawKernel和ElementwiseKernel编写高性能代码
- **Python生态集成**：易于与可视化、数据分析工具集成
- **快速原型开发**：比纯CUDA C++开发快速得多

### 1.2 技术栈

```
Python 3.8+
├── CuPy 12.0+          # GPU加速计算
├── NumPy               # CPU数组操作
├── Numba (可选)        # JIT编译优化
├── Matplotlib          # 性能可视化
├── Pygame/ModernGL     # 3D渲染
└── OpenCV              # 视频输出
```

---

## 2. 数据结构设计

### 2.1 刚体数据结构（GPU）

```python
import cupy as cp
import numpy as np

class RigidBodySystem:
    """GPU上的刚体物理系统"""
    
    def __init__(self, num_objects, device_id=0):
        """
        初始化刚体系统
        
        Args:
            num_objects: 物体数量
            device_id: GPU设备ID
        """
        with cp.cuda.Device(device_id):
            # 位置 [N, 3]
            self.positions = cp.zeros((num_objects, 3), dtype=cp.float32)
            
            # 速度 [N, 3]
            self.velocities = cp.zeros((num_objects, 3), dtype=cp.float32)
            
            # 加速度 [N, 3]
            self.accelerations = cp.zeros((num_objects, 3), dtype=cp.float32)
            
            # 力累积 [N, 3]
            self.forces = cp.zeros((num_objects, 3), dtype=cp.float32)
            
            # 标量属性 [N]
            self.radii = cp.ones(num_objects, dtype=cp.float32) * 0.5
            self.masses = cp.ones(num_objects, dtype=cp.float32) * 1.0
            self.restitutions = cp.ones(num_objects, dtype=cp.float32) * 0.8
            self.frictions = cp.ones(num_objects, dtype=cp.float32) * 0.3
            
            # 网格索引 [N]
            self.grid_indices = cp.zeros(num_objects, dtype=cp.int32)
            
            # 状态标记 [N]
            self.is_sleeping = cp.zeros(num_objects, dtype=cp.bool_)
            
            # 颜色 [N, 3] (可选，用于渲染)
            self.colors = cp.random.rand(num_objects, 3).astype(cp.float32)
            
            self.num_objects = num_objects
            self.device_id = device_id
    
    def to_cpu(self):
        """将数据传输到CPU（用于渲染或分析）"""
        return {
            'positions': cp.asnumpy(self.positions),
            'velocities': cp.asnumpy(self.velocities),
            'radii': cp.asnumpy(self.radii),
            'colors': cp.asnumpy(self.colors)
        }
```

### 2.2 Uniform Grid结构（GPU）

```python
class UniformGrid:
    """GPU上的均匀网格空间分割结构"""
    
    def __init__(self, world_min, world_max, cell_size, device_id=0):
        """
        初始化均匀网格
        
        Args:
            world_min: 世界空间最小坐标 [3]
            world_max: 世界空间最大坐标 [3]
            cell_size: 网格单元大小
            device_id: GPU设备ID
        """
        with cp.cuda.Device(device_id):
            self.world_min = cp.array(world_min, dtype=cp.float32)
            self.world_max = cp.array(world_max, dtype=cp.float32)
            self.cell_size = float(cell_size)
            
            # 计算网格分辨率
            world_size = self.world_max - self.world_min
            self.resolution = cp.ceil(world_size / cell_size).astype(cp.int32)
            self.total_cells = int(cp.prod(self.resolution))
            
            # 网格单元计数 [total_cells]
            self.cell_counts = cp.zeros(self.total_cells, dtype=cp.int32)
            
            # 网格单元起始索引 [total_cells]
            self.cell_starts = cp.zeros(self.total_cells, dtype=cp.int32)
            
            # 物体到网格的映射（排序后）
            self.sorted_indices = None
            self.sorted_grid_hashes = None
            
            self.device_id = device_id
    
    def get_grid_coord(self, positions):
        """
        将世界坐标转换为网格坐标
        
        Args:
            positions: 世界坐标 [N, 3]
        
        Returns:
            grid_coords: 网格坐标 [N, 3]
        """
        normalized = (positions - self.world_min) / self.cell_size
        grid_coords = cp.floor(normalized).astype(cp.int32)
        
        # 边界夹紧
        grid_coords = cp.clip(grid_coords, 0, self.resolution - 1)
        return grid_coords
    
    def get_grid_hash(self, grid_coords):
        """
        将3D网格坐标转换为1D哈希值
        
        Args:
            grid_coords: 网格坐标 [N, 3]
        
        Returns:
            hashes: 1D哈希值 [N]
        """
        return (grid_coords[:, 2] * self.resolution[1] * self.resolution[0] +
                grid_coords[:, 1] * self.resolution[0] +
                grid_coords[:, 0])
```

---

## 3. CUDA Kernel实现

### 3.1 网格构建Kernel

```python
# CUDA核函数：计算网格哈希
grid_hash_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_grid_hash(
    const float* positions,     // [N, 3]
    int* grid_hashes,           // [N]
    const float* world_min,     // [3]
    float cell_size,
    const int* resolution,      // [3]
    int num_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // 计算网格坐标
    int gx = (int)((positions[idx * 3 + 0] - world_min[0]) / cell_size);
    int gy = (int)((positions[idx * 3 + 1] - world_min[1]) / cell_size);
    int gz = (int)((positions[idx * 3 + 2] - world_min[2]) / cell_size);
    
    // 边界夹紧
    gx = max(0, min(gx, resolution[0] - 1));
    gy = max(0, min(gy, resolution[1] - 1));
    gz = max(0, min(gz, resolution[2] - 1));
    
    // 计算1D哈希
    grid_hashes[idx] = gz * resolution[1] * resolution[0] + 
                       gy * resolution[0] + gx;
}
''', 'compute_grid_hash')


# CUDA核函数：重排数据
reorder_data_kernel = cp.RawKernel(r'''
extern "C" __global__
void reorder_data(
    const float* positions_in,      // [N, 3]
    const float* velocities_in,     // [N, 3]
    const float* radii_in,          // [N]
    float* positions_out,           // [N, 3]
    float* velocities_out,          // [N, 3]
    float* radii_out,               // [N]
    const int* sorted_indices,      // [N]
    int num_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    int original_idx = sorted_indices[idx];
    
    // 重排位置
    positions_out[idx * 3 + 0] = positions_in[original_idx * 3 + 0];
    positions_out[idx * 3 + 1] = positions_in[original_idx * 3 + 1];
    positions_out[idx * 3 + 2] = positions_in[original_idx * 3 + 2];
    
    // 重排速度
    velocities_out[idx * 3 + 0] = velocities_in[original_idx * 3 + 0];
    velocities_out[idx * 3 + 1] = velocities_in[original_idx * 3 + 1];
    velocities_out[idx * 3 + 2] = velocities_in[original_idx * 3 + 2];
    
    // 重排半径
    radii_out[idx] = radii_in[original_idx];
}
''', 'reorder_data')


# CUDA核函数：查找单元起始位置
find_cell_start_kernel = cp.RawKernel(r'''
extern "C" __global__
void find_cell_start(
    const int* sorted_hashes,   // [N]
    int* cell_starts,           // [total_cells]
    int* cell_ends,             // [total_cells]
    int num_objects,
    int total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    int hash = sorted_hashes[idx];
    
    // 检查是否是该单元的第一个元素
    if (idx == 0 || sorted_hashes[idx - 1] != hash) {
        cell_starts[hash] = idx;
    }
    
    // 检查是否是该单元的最后一个元素
    if (idx == num_objects - 1 || sorted_hashes[idx + 1] != hash) {
        cell_ends[hash] = idx + 1;
    }
}
''', 'find_cell_start')
```

### 3.2 碰撞检测Kernel

```python
# CUDA核函数：Broad Phase碰撞检测
broad_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void broad_phase_collision(
    const float* positions,         // [N, 3]
    const float* radii,             // [N]
    const int* cell_starts,         // [total_cells]
    const int* cell_ends,           // [total_cells]
    const int* resolution,          // [3]
    float cell_size,
    const float* world_min,         // [3]
    int* collision_pairs,           // [max_pairs, 2] 输出
    int* pair_count,                // [1] 原子计数
    int num_objects,
    int max_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // 当前物体的位置和半径
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    float r1 = radii[idx];
    
    // 计算所在网格
    int gx = (int)((px - world_min[0]) / cell_size);
    int gy = (int)((py - world_min[1]) / cell_size);
    int gz = (int)((pz - world_min[2]) / cell_size);
    
    // 遍历周围27个单元（3x3x3）
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = gx + dx;
                int ny = gy + dy;
                int nz = gz + dz;
                
                // 边界检查
                if (nx < 0 || nx >= resolution[0] ||
                    ny < 0 || ny >= resolution[1] ||
                    nz < 0 || nz >= resolution[2]) {
                    continue;
                }
                
                // 计算邻居单元哈希
                int cell_hash = nz * resolution[1] * resolution[0] +
                                ny * resolution[0] + nx;
                
                int start = cell_starts[cell_hash];
                int end = cell_ends[cell_hash];
                
                // 遍历该单元中的所有物体
                for (int j = start; j < end; j++) {
                    if (j <= idx) continue;  // 避免重复检测和自碰撞
                    
                    float qx = positions[j * 3 + 0];
                    float qy = positions[j * 3 + 1];
                    float qz = positions[j * 3 + 2];
                    float r2 = radii[j];
                    
                    // AABB粗检测
                    float dx_dist = px - qx;
                    float dy_dist = py - qy;
                    float dz_dist = pz - qz;
                    float dist_sq = dx_dist * dx_dist + 
                                    dy_dist * dy_dist + 
                                    dz_dist * dz_dist;
                    
                    float sum_r = r1 + r2;
                    
                    // 球体相交测试
                    if (dist_sq < sum_r * sum_r) {
                        // 记录碰撞对
                        int pair_idx = atomicAdd(pair_count, 1);
                        if (pair_idx < max_pairs) {
                            collision_pairs[pair_idx * 2 + 0] = idx;
                            collision_pairs[pair_idx * 2 + 1] = j;
                        }
                    }
                }
            }
        }
    }
}
''', 'broad_phase_collision')


# CUDA核函数：碰撞响应
collision_response_kernel = cp.RawKernel(r'''
extern "C" __global__
void resolve_collisions(
    float* positions,               // [N, 3]
    float* velocities,              // [N, 3]
    const float* radii,             // [N]
    const float* masses,            // [N]
    const float* restitutions,      // [N]
    const int* collision_pairs,     // [num_pairs, 2]
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    int id_a = collision_pairs[idx * 2 + 0];
    int id_b = collision_pairs[idx * 2 + 1];
    
    // 读取数据
    float3 pos_a = make_float3(positions[id_a * 3 + 0],
                                positions[id_a * 3 + 1],
                                positions[id_a * 3 + 2]);
    float3 pos_b = make_float3(positions[id_b * 3 + 0],
                                positions[id_b * 3 + 1],
                                positions[id_b * 3 + 2]);
    
    float3 vel_a = make_float3(velocities[id_a * 3 + 0],
                                velocities[id_a * 3 + 1],
                                velocities[id_a * 3 + 2]);
    float3 vel_b = make_float3(velocities[id_b * 3 + 0],
                                velocities[id_b * 3 + 1],
                                velocities[id_b * 3 + 2]);
    
    float r_a = radii[id_a];
    float r_b = radii[id_b];
    float m_a = masses[id_a];
    float m_b = masses[id_b];
    float e = min(restitutions[id_a], restitutions[id_b]);
    
    // 计算碰撞法线
    float3 delta = make_float3(pos_b.x - pos_a.x,
                                pos_b.y - pos_a.y,
                                pos_b.z - pos_a.z);
    float dist = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    
    if (dist < 1e-6f) return;  // 避免除零
    
    float3 normal = make_float3(delta.x / dist, delta.y / dist, delta.z / dist);
    
    // 相对速度
    float3 rel_vel = make_float3(vel_b.x - vel_a.x,
                                  vel_b.y - vel_a.y,
                                  vel_b.z - vel_a.z);
    float vel_along_normal = rel_vel.x * normal.x +
                              rel_vel.y * normal.y +
                              rel_vel.z * normal.z;
    
    // 物体正在分离
    if (vel_along_normal > 0) return;
    
    // 计算冲量
    float j = -(1.0f + e) * vel_along_normal;
    j /= (1.0f / m_a + 1.0f / m_b);
    
    float3 impulse = make_float3(j * normal.x, j * normal.y, j * normal.z);
    
    // 应用冲量（使用原子操作避免竞争条件）
    atomicAdd(&velocities[id_a * 3 + 0], -impulse.x / m_a);
    atomicAdd(&velocities[id_a * 3 + 1], -impulse.y / m_a);
    atomicAdd(&velocities[id_a * 3 + 2], -impulse.z / m_a);
    
    atomicAdd(&velocities[id_b * 3 + 0], impulse.x / m_b);
    atomicAdd(&velocities[id_b * 3 + 1], impulse.y / m_b);
    atomicAdd(&velocities[id_b * 3 + 2], impulse.z / m_b);
    
    // 位置修正（避免穿透）
    float penetration = (r_a + r_b) - dist;
    if (penetration > 0) {
        float correction_mag = penetration * 0.5f;  // 各修正一半
        
        float3 correction = make_float3(normal.x * correction_mag,
                                         normal.y * correction_mag,
                                         normal.z * correction_mag);
        
        atomicAdd(&positions[id_a * 3 + 0], -correction.x);
        atomicAdd(&positions[id_a * 3 + 1], -correction.y);
        atomicAdd(&positions[id_a * 3 + 2], -correction.z);
        
        atomicAdd(&positions[id_b * 3 + 0], correction.x);
        atomicAdd(&positions[id_b * 3 + 1], correction.y);
        atomicAdd(&positions[id_b * 3 + 2], correction.z);
    }
}
''', 'resolve_collisions')
```

### 3.3 物理积分Kernel

```python
# CUDA核函数：半隐式Euler积分
integrate_kernel = cp.RawKernel(r'''
extern "C" __global__
void integrate_physics(
    float* positions,       // [N, 3]
    float* velocities,      // [N, 3]
    const float* forces,    // [N, 3]
    const float* masses,    // [N]
    const float* gravity,   // [3]
    float dt,
    float damping,
    const float* bounds_min,    // [3]
    const float* bounds_max,    // [3]
    const float* radii,         // [N]
    const float* restitutions,  // [N]
    int num_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    float mass = masses[idx];
    float inv_mass = 1.0f / mass;
    float radius = radii[idx];
    float restitution = restitutions[idx];
    
    // 计算加速度
    float ax = (forces[idx * 3 + 0] * inv_mass) + gravity[0];
    float ay = (forces[idx * 3 + 1] * inv_mass) + gravity[1];
    float az = (forces[idx * 3 + 2] * inv_mass) + gravity[2];
    
    // 更新速度（半隐式Euler）
    velocities[idx * 3 + 0] += ax * dt;
    velocities[idx * 3 + 1] += ay * dt;
    velocities[idx * 3 + 2] += az * dt;
    
    // 应用阻尼
    float damping_factor = 1.0f - damping * dt;
    velocities[idx * 3 + 0] *= damping_factor;
    velocities[idx * 3 + 1] *= damping_factor;
    velocities[idx * 3 + 2] *= damping_factor;
    
    // 更新位置
    positions[idx * 3 + 0] += velocities[idx * 3 + 0] * dt;
    positions[idx * 3 + 1] += velocities[idx * 3 + 1] * dt;
    positions[idx * 3 + 2] += velocities[idx * 3 + 2] * dt;
    
    // 边界碰撞处理
    for (int axis = 0; axis < 3; axis++) {
        float pos = positions[idx * 3 + axis];
        float vel = velocities[idx * 3 + axis];
        
        // 下边界
        if (pos - radius < bounds_min[axis]) {
            positions[idx * 3 + axis] = bounds_min[axis] + radius;
            velocities[idx * 3 + axis] = -vel * restitution;
        }
        
        // 上边界
        if (pos + radius > bounds_max[axis]) {
            positions[idx * 3 + axis] = bounds_max[axis] - radius;
            velocities[idx * 3 + axis] = -vel * restitution;
        }
    }
}
''', 'integrate_physics')
```

---

## 4. 主仿真循环

### 4.1 PhysicsSimulator类

```python
class PhysicsSimulator:
    """基于CuPy的GPU物理仿真器"""
    
    def __init__(self, num_objects, world_bounds, cell_size=2.0, device_id=0):
        """
        初始化物理仿真器
        
        Args:
            num_objects: 物体数量
            world_bounds: 世界边界 [(xmin, ymin, zmin), (xmax, ymax, zmax)]
            cell_size: 网格单元大小
            device_id: GPU设备ID
        """
        self.device_id = device_id
        
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
            self.gravity = cp.array([0.0, -9.8, 0.0], dtype=cp.float32)
            self.damping = 0.01
            self.dt = 1.0 / 60.0  # 60 FPS
            
            # 碰撞对缓冲区
            self.max_pairs = num_objects * 50  # 假设平均每个物体50个潜在碰撞
            self.collision_pairs = cp.zeros((self.max_pairs, 2), dtype=cp.int32)
            self.pair_count = cp.zeros(1, dtype=cp.int32)
            
            # 临时缓冲区
            self.sorted_positions = cp.zeros_like(self.bodies.positions)
            self.sorted_velocities = cp.zeros_like(self.bodies.velocities)
            self.sorted_radii = cp.zeros_like(self.bodies.radii)
            
            # CUDA流（用于异步操作）
            self.stream = cp.cuda.Stream()
            
            # 性能统计
            self.stats = {
                'grid_build_time': 0.0,
                'collision_detect_time': 0.0,
                'collision_resolve_time': 0.0,
                'integrate_time': 0.0
            }
    
    def build_grid(self):
        """构建空间网格"""
        with cp.cuda.Device(self.device_id):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            
            # 1. 计算网格哈希
            grid_coords = self.grid.get_grid_coord(self.bodies.positions)
            grid_hashes = self.grid.get_grid_hash(grid_coords)
            
            # 2. 排序（按哈希值）
            sorted_indices = cp.argsort(grid_hashes)
            sorted_hashes = grid_hashes[sorted_indices]
            
            # 3. 重排数据（提高缓存局部性）
            threads_per_block = 256
            blocks = (self.bodies.num_objects + threads_per_block - 1) // threads_per_block
            
            reorder_data_kernel(
                (blocks,), (threads_per_block,),
                (self.bodies.positions, self.bodies.velocities, self.bodies.radii,
                 self.sorted_positions, self.sorted_velocities, self.sorted_radii,
                 sorted_indices, self.bodies.num_objects)
            )
            
            # 4. 查找每个单元的起始和结束位置
            self.grid.cell_starts.fill(-1)
            cell_ends = cp.zeros_like(self.grid.cell_starts)
            
            find_cell_start_kernel(
                (blocks,), (threads_per_block,),
                (sorted_hashes, self.grid.cell_starts, cell_ends,
                 self.bodies.num_objects, self.grid.total_cells)
            )
            
            # 将-1替换为0（表示空单元）
            self.grid.cell_starts[self.grid.cell_starts == -1] = 0
            
            end.record()
            end.synchronize()
            self.stats['grid_build_time'] = cp.cuda.get_elapsed_time(start, end)
    
    def detect_collisions(self):
        """检测碰撞"""
        with cp.cuda.Device(self.device_id):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            
            # 重置碰撞对计数
            self.pair_count.fill(0)
            
            # 执行Broad Phase检测
            threads_per_block = 256
            blocks = (self.bodies.num_objects + threads_per_block - 1) // threads_per_block
            
            broad_phase_kernel(
                (blocks,), (threads_per_block,),
                (self.sorted_positions, self.sorted_radii,
                 self.grid.cell_starts, cp.zeros_like(self.grid.cell_starts),  # cell_ends
                 self.grid.resolution, self.grid.cell_size, self.grid.world_min,
                 self.collision_pairs, self.pair_count,
                 self.bodies.num_objects, self.max_pairs)
            )
            
            end.record()
            end.synchronize()
            self.stats['collision_detect_time'] = cp.cuda.get_elapsed_time(start, end)
            
            return int(self.pair_count[0])
    
    def resolve_collisions(self, num_pairs):
        """解决碰撞"""
        if num_pairs == 0:
            return
        
        with cp.cuda.Device(self.device_id):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            
            threads_per_block = 256
            blocks = (num_pairs + threads_per_block - 1) // threads_per_block
            
            collision_response_kernel(
                (blocks,), (threads_per_block,),
                (self.bodies.positions, self.bodies.velocities,
                 self.bodies.radii, self.bodies.masses, self.bodies.restitutions,
                 self.collision_pairs, num_pairs)
            )
            
            end.record()
            end.synchronize()
            self.stats['collision_resolve_time'] = cp.cuda.get_elapsed_time(start, end)
    
    def integrate(self):
        """物理积分"""
        with cp.cuda.Device(self.device_id):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            
            threads_per_block = 256
            blocks = (self.bodies.num_objects + threads_per_block - 1) // threads_per_block
            
            integrate_kernel(
                (blocks,), (threads_per_block,),
                (self.bodies.positions, self.bodies.velocities, self.bodies.forces,
                 self.bodies.masses, self.gravity, self.dt, self.damping,
                 self.grid.world_min, self.grid.world_max,
                 self.bodies.radii, self.bodies.restitutions,
                 self.bodies.num_objects)
            )
            
            # 清空力累积
            self.bodies.forces.fill(0)
            
            end.record()
            end.synchronize()
            self.stats['integrate_time'] = cp.cuda.get_elapsed_time(start, end)
    
    def step(self):
        """执行一个仿真步"""
        # 1. 构建空间网格
        self.build_grid()
        
        # 2. 检测碰撞
        num_pairs = self.detect_collisions()
        
        # 3. 解决碰撞
        self.resolve_collisions(num_pairs)
        
        # 4. 物理积分
        self.integrate()
        
        return num_pairs
    
    def get_stats(self):
        """获取性能统计"""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total_time': total,
            'fps': 1000.0 / total if total > 0 else 0.0
        }
```

### 4.2 主循环示例

```python
def main():
    """主仿真循环"""
    
    # 配置
    NUM_OBJECTS = 10000
    WORLD_BOUNDS = [(-50, -50, -50), (50, 50, 50)]
    CELL_SIZE = 2.0
    NUM_FRAMES = 1000
    
    # 初始化仿真器
    sim = PhysicsSimulator(NUM_OBJECTS, WORLD_BOUNDS, CELL_SIZE)
    
    # 随机初始化物体
    with cp.cuda.Device(sim.device_id):
        # 位置：随机分布在上半空间
        sim.bodies.positions[:, 0] = cp.random.uniform(-40, 40, NUM_OBJECTS)
        sim.bodies.positions[:, 1] = cp.random.uniform(20, 40, NUM_OBJECTS)
        sim.bodies.positions[:, 2] = cp.random.uniform(-40, 40, NUM_OBJECTS)
        
        # 速度：随机初速度
        sim.bodies.velocities = cp.random.uniform(-5, 5, (NUM_OBJECTS, 3))
        
        # 半径：随机大小
        sim.bodies.radii = cp.random.uniform(0.3, 0.8, NUM_OBJECTS)
        
        # 质量：与体积成正比
        sim.bodies.masses = sim.bodies.radii ** 3 * 1000.0
        
        # 弹性系数
        sim.bodies.restitutions = cp.random.uniform(0.6, 0.9, NUM_OBJECTS)
    
    # 仿真循环
    print("Starting simulation...")
    frame_data = []
    
    for frame in range(NUM_FRAMES):
        # 执行仿真步
        num_collisions = sim.step()
        
        # 获取性能统计
        if frame % 60 == 0:
            stats = sim.get_stats()
            print(f"Frame {frame}: FPS={stats['fps']:.1f}, "
                  f"Collisions={num_collisions}, "
                  f"Grid={stats['grid_build_time']:.2f}ms, "
                  f"Detect={stats['collision_detect_time']:.2f}ms, "
                  f"Resolve={stats['collision_resolve_time']:.2f}ms, "
                  f"Integrate={stats['integrate_time']:.2f}ms")
        
        # 保存帧数据（用于渲染或分析）
        if frame % 2 == 0:  # 每2帧保存一次
            data = sim.bodies.to_cpu()
            frame_data.append(data)
    
    print("Simulation complete!")
    
    # 导出数据
    import pickle
    with open('simulation_data.pkl', 'wb') as f:
        pickle.dump(frame_data, f)
    
    return frame_data


if __name__ == '__main__':
    main()
```

---

## 5. 3D Gaussians碰撞检测

### 5.1 Gaussian数据结构

```python
class GaussianScene:
    """3D Gaussian Splatting场景"""
    
    def __init__(self, num_gaussians, device_id=0):
        with cp.cuda.Device(device_id):
            # Gaussian中心位置 [N, 3]
            self.positions = cp.zeros((num_gaussians, 3), dtype=cp.float32)
            
            # 缩放参数 [N, 3]
            self.scales = cp.ones((num_gaussians, 3), dtype=cp.float32) * 0.5
            
            # 旋转四元数 [N, 4]
            self.rotations = cp.zeros((num_gaussians, 4), dtype=cp.float32)
            self.rotations[:, 3] = 1.0  # 单位四元数
            
            # 不透明度 [N]
            self.opacities = cp.ones(num_gaussians, dtype=cp.float32) * 0.8
            
            # 颜色（SH系数） [N, 3]
            self.colors = cp.random.rand(num_gaussians, 3).astype(cp.float32)
            
            self.num_gaussians = num_gaussians
            self.device_id = device_id
    
    @staticmethod
    def load_from_ply(ply_file, device_id=0):
        """从PLY文件加载Gaussian场景"""
        # 这里需要实现PLY文件读取
        # 参考3D Gaussian Splatting原始代码
        pass
```

### 5.2 Gaussian-Sphere碰撞Kernel

```python
# CUDA核函数：Gaussian-Sphere碰撞检测
gaussian_collision_kernel = cp.RawKernel(r'''
extern "C" __global__
void detect_gaussian_collisions(
    const float* sphere_positions,      // [N, 3]
    const float* sphere_radii,          // [N]
    const float* gaussian_positions,    // [M, 3]
    const float* gaussian_scales,       // [M, 3]
    const float* gaussian_rotations,    // [M, 4] (quaternions)
    const float* gaussian_opacities,    // [M]
    float* collision_forces,            // [N, 3] 输出
    float density_threshold,
    int num_spheres,
    int num_gaussians
) {
    int sphere_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sphere_idx >= num_spheres) return;
    
    float3 sphere_pos = make_float3(
        sphere_positions[sphere_idx * 3 + 0],
        sphere_positions[sphere_idx * 3 + 1],
        sphere_positions[sphere_idx * 3 + 2]
    );
    float sphere_r = sphere_radii[sphere_idx];
    
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    float total_density = 0.0f;
    
    // 遍历所有Gaussians
    for (int g = 0; g < num_gaussians; g++) {
        float3 gauss_pos = make_float3(
            gaussian_positions[g * 3 + 0],
            gaussian_positions[g * 3 + 1],
            gaussian_positions[g * 3 + 2]
        );
        
        // 快速距离剔除
        float3 delta = make_float3(
            sphere_pos.x - gauss_pos.x,
            sphere_pos.y - gauss_pos.y,
            sphere_pos.z - gauss_pos.z
        );
        float dist_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        
        float max_scale = fmaxf(gaussian_scales[g * 3 + 0],
                         fmaxf(gaussian_scales[g * 3 + 1],
                               gaussian_scales[g * 3 + 2]));
        float cutoff = (max_scale * 3.0f + sphere_r);  // 3-sigma截断
        
        if (dist_sq > cutoff * cutoff) continue;
        
        // 旋转矩阵（从四元数）
        float4 q = make_float4(
            gaussian_rotations[g * 4 + 0],
            gaussian_rotations[g * 4 + 1],
            gaussian_rotations[g * 4 + 2],
            gaussian_rotations[g * 4 + 3]
        );
        
        // 简化的旋转变换（完整版需要构建3x3矩阵）
        // 这里假设Gaussian对齐到坐标轴
        float3 scale = make_float3(
            gaussian_scales[g * 3 + 0],
            gaussian_scales[g * 3 + 1],
            gaussian_scales[g * 3 + 2]
        );
        
        // 局部坐标
        float3 local_pos = make_float3(
            delta.x / scale.x,
            delta.y / scale.y,
            delta.z / scale.z
        );
        
        // Gaussian密度
        float exponent = -0.5f * (local_pos.x * local_pos.x +
                                   local_pos.y * local_pos.y +
                                   local_pos.z * local_pos.z);
        float density = gaussian_opacities[g] * expf(exponent);
        
        total_density += density;
        
        // 如果穿透，计算斥力
        if (density > density_threshold) {
            // 斥力方向：远离Gaussian中心
            float force_mag = density * 100.0f;  // 力的强度
            float dist = sqrtf(dist_sq);
            
            if (dist > 1e-6f) {
                float3 force_dir = make_float3(
                    delta.x / dist,
                    delta.y / dist,
                    delta.z / dist
                );
                
                total_force.x += force_dir.x * force_mag;
                total_force.y += force_dir.y * force_mag;
                total_force.z += force_dir.z * force_mag;
            }
        }
    }
    
    // 写入结果
    collision_forces[sphere_idx * 3 + 0] = total_force.x;
    collision_forces[sphere_idx * 3 + 1] = total_force.y;
    collision_forces[sphere_idx * 3 + 2] = total_force.z;
}
''', 'detect_gaussian_collisions')
```

### 5.3 混合仿真

```python
class HybridSimulator(PhysicsSimulator):
    """支持Gaussian场景的混合仿真器"""
    
    def __init__(self, num_objects, world_bounds, gaussian_scene, cell_size=2.0, device_id=0):
        super().__init__(num_objects, world_bounds, cell_size, device_id)
        
        self.gaussian_scene = gaussian_scene
        self.gaussian_forces = cp.zeros((num_objects, 3), dtype=cp.float32)
        self.density_threshold = 0.1
    
    def detect_gaussian_collisions(self):
        """检测与Gaussian场景的碰撞"""
        with cp.cuda.Device(self.device_id):
            threads_per_block = 256
            blocks = (self.bodies.num_objects + threads_per_block - 1) // threads_per_block
            
            gaussian_collision_kernel(
                (blocks,), (threads_per_block,),
                (self.bodies.positions, self.bodies.radii,
                 self.gaussian_scene.positions, self.gaussian_scene.scales,
                 self.gaussian_scene.rotations, self.gaussian_scene.opacities,
                 self.gaussian_forces, self.density_threshold,
                 self.bodies.num_objects, self.gaussian_scene.num_gaussians)
            )
            
            # 将力添加到物体
            self.bodies.forces += self.gaussian_forces
    
    def step(self):
        """执行一个混合仿真步"""
        # 1. 构建空间网格
        self.build_grid()
        
        # 2. 刚体间碰撞检测
        num_pairs = self.detect_collisions()
        
        # 3. Gaussian碰撞检测
        self.detect_gaussian_collisions()
        
        # 4. 解决碰撞
        self.resolve_collisions(num_pairs)
        
        # 5. 物理积分
        self.integrate()
        
        return num_pairs
```

---

## 6. 性能分析工具

### 6.1 性能监控

```python
class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.events = {}
    
    def start_event(self, name):
        """开始计时事件"""
        event = cp.cuda.Event()
        event.record()
        self.events[name] = event
    
    def end_event(self, name):
        """结束计时事件"""
        if name not in self.events:
            return
        
        end_event = cp.cuda.Event()
        end_event.record()
        end_event.synchronize()
        
        elapsed = cp.cuda.get_elapsed_time(self.events[name], end_event)
        self.metrics[name].append(elapsed)
        del self.events[name]
    
    def get_statistics(self):
        """获取统计数据"""
        stats = {}
        for name, values in self.metrics.items():
            values_array = np.array(values)
            stats[name] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'median': np.median(values_array)
            }
        return stats
    
    def plot_timeline(self, save_path='performance_timeline.png'):
        """绘制性能时间线"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 8))
        if len(self.metrics) == 1:
            axes = [axes]
        
        for ax, (name, values) in zip(axes, self.metrics.items()):
            ax.plot(values)
            ax.set_ylabel(f'{name} (ms)')
            ax.grid(True)
        
        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
```

### 6.2 规模测试

```python
def benchmark_scaling():
    """测试不同规模下的性能"""
    
    scales = [1000, 5000, 10000, 50000, 100000]
    results = []
    
    for num_objects in scales:
        print(f"\nTesting with {num_objects} objects...")
        
        sim = PhysicsSimulator(
            num_objects,
            [(-50, -50, -50), (50, 50, 50)],
            cell_size=2.0
        )
        
        # 初始化
        with cp.cuda.Device(sim.device_id):
            sim.bodies.positions = cp.random.uniform(-40, 40, (num_objects, 3))
            sim.bodies.velocities = cp.random.uniform(-5, 5, (num_objects, 3))
            sim.bodies.radii = cp.random.uniform(0.3, 0.8, num_objects)
            sim.bodies.masses = sim.bodies.radii ** 3 * 1000.0
        
        # 预热
        for _ in range(10):
            sim.step()
        
        # 测试
        times = []
        for _ in range(100):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            
            start.record()
            sim.step()
            end.record()
            end.synchronize()
            
            times.append(cp.cuda.get_elapsed_time(start, end))
        
        avg_time = np.mean(times)
        fps = 1000.0 / avg_time
        
        results.append({
            'num_objects': num_objects,
            'avg_time_ms': avg_time,
            'fps': fps,
            'std_time_ms': np.std(times)
        })
        
        print(f"  Avg time: {avg_time:.2f}ms, FPS: {fps:.1f}")
    
    # 绘制结果
    import matplotlib.pyplot as plt
    
    nums = [r['num_objects'] for r in results]
    times = [r['avg_time_ms'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(nums, times, 'o-')
    plt.xlabel('Number of Objects')
    plt.ylabel('Average Frame Time (ms)')
    plt.title('Performance Scaling')
    plt.grid(True)
    plt.savefig('scaling_benchmark.png')
    
    return results
```

---

## 7. 可视化与动画导出

### 7.1 使用Matplotlib实时可视化

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RealtimeVisualizer:
    """实时3D可视化"""
    
    def __init__(self, world_bounds):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlim(world_bounds[0][0], world_bounds[1][0])
        self.ax.set_ylim(world_bounds[0][1], world_bounds[1][1])
        self.ax.set_zlim(world_bounds[0][2], world_bounds[1][2])
        
        self.scatter = None
    
    def update(self, positions, colors):
        """更新显示"""
        if self.scatter is not None:
            self.scatter.remove()
        
        self.scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            s=20
        )
        
        plt.draw()
        plt.pause(0.001)
```

### 7.2 导出视频

```python
import cv2

class VideoExporter:
    """视频导出器"""
    
    def __init__(self, filename, fps=60, resolution=(1920, 1080)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
        self.resolution = resolution
    
    def add_frame_from_matplotlib(self, fig):
        """从matplotlib图形添加帧"""
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, self.resolution)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.writer.write(img)
    
    def release(self):
        """完成导出"""
        self.writer.release()
```

---

## 8. 完整示例：重力下落场景

```python
def create_gravity_fall_animation():
    """创建重力下落动画"""
    
    # 配置
    NUM_OBJECTS = 5000
    WORLD_BOUNDS = [(-30, 0, -30), (30, 60, 30)]
    NUM_FRAMES = 600  # 10秒 @ 60 FPS
    
    # 初始化
    sim = PhysicsSimulator(NUM_OBJECTS, WORLD_BOUNDS, cell_size=2.5)
    visualizer = RealtimeVisualizer(WORLD_BOUNDS)
    video = VideoExporter('gravity_fall.mp4', fps=60)
    
    # 场景设置：物体从上方随机位置下落
    with cp.cuda.Device(sim.device_id):
        # 在上半空间随机分布
        sim.bodies.positions[:, 0] = cp.random.uniform(-25, 25, NUM_OBJECTS)
        sim.bodies.positions[:, 1] = cp.random.uniform(40, 55, NUM_OBJECTS)
        sim.bodies.positions[:, 2] = cp.random.uniform(-25, 25, NUM_OBJECTS)
        
        # 小的随机初速度
        sim.bodies.velocities = cp.random.uniform(-2, 2, (NUM_OBJECTS, 3))
        
        # 随机半径
        sim.bodies.radii = cp.random.uniform(0.2, 0.6, NUM_OBJECTS)
        sim.bodies.masses = sim.bodies.radii ** 3 * 1000.0
        
        # 高弹性
        sim.bodies.restitutions = cp.random.uniform(0.7, 0.95, NUM_OBJECTS)
    
    # 仿真循环
    print("Simulating and rendering...")
    
    for frame in range(NUM_FRAMES):
        # 仿真步
        num_collisions = sim.step()
        
        # 获取CPU数据
        data = sim.bodies.to_cpu()
        
        # 更新可视化
        visualizer.update(data['positions'], data['colors'])
        
        # 保存帧
        video.add_frame_from_matplotlib(visualizer.fig)
        
        # 打印进度
        if frame % 60 == 0:
            stats = sim.get_stats()
            print(f"Frame {frame}/{NUM_FRAMES}: "
                  f"FPS={stats['fps']:.1f}, "
                  f"Collisions={num_collisions}")
    
    # 完成
    video.release()
    plt.close()
    
    print("Animation saved to gravity_fall.mp4")


if __name__ == '__main__':
    create_gravity_fall_animation()
```

---

## 9. 总结与优化建议

### 9.1 CuPy方案的优势

1. **开发效率高**：Python语法，NumPy兼容
2. **易于调试**：可以轻松在CPU/GPU间切换
3. **生态系统丰富**：集成Matplotlib、OpenCV等工具
4. **性能优秀**：接近纯CUDA C++的性能

### 9.2 性能优化清单

- [ ] **内存优化**
  - 使用连续内存布局（SOA）
  - 减少CPU-GPU数据传输
  - 使用内存池（CuPy默认支持）

- [ ] **计算优化**
  - 调整线程块大小（通常256或512）
  - 使用共享内存减少全局内存访问
  - 避免过多的原子操作

- [ ] **算法优化**
  - 动态调整网格大小
  - 实现休眠机制（静止物体跳过计算）
  - 使用更高效的排序算法（Radix Sort）

- [ ] **并行优化**
  - 使用CUDA流实现异步计算
  - Pipeline化不同阶段
  - 多GPU支持（使用CuPy的多设备API）

### 9.3 扩展方向

1. **更复杂的形状**：支持盒子、胶囊体等
2. **约束系统**：铰链、弹簧等
3. **软体物理**：变形物体
4. **流体模拟**：SPH方法
5. **机器学习集成**：学习物理参数

### 9.4 与OpenGL方案对比

| 特性 | CuPy (CUDA) | OpenGL Compute Shader |
|------|-------------|------------------------|
| 开发效率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 计算性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 渲染集成 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 跨平台 | NVIDIA only | 广泛支持 |
| 调试难度 | 低 | 中 |
| 生态系统 | Python | C++ |

---

## 10. 参考代码结构

```
project/
├── src/
│   ├── __init__.py
│   ├── rigid_body.py          # 刚体系统
│   ├── spatial_grid.py        # 空间网格
│   ├── kernels.py             # CUDA Kernels
│   ├── simulator.py           # 主仿真器
│   ├── gaussian_scene.py      # Gaussian场景
│   └── visualizer.py          # 可视化工具
├── tests/
│   ├── test_collision.py      # 碰撞检测测试
│   ├── test_physics.py        # 物理仿真测试
│   └── benchmark.py           # 性能测试
├── examples/
│   ├── gravity_fall.py        # 重力下落示例
│   ├── explosion.py           # 爆炸场景
│   └── gaussian_hybrid.py     # Gaussian混合场景
├── requirements.txt
└── README.md
```

---

**完成时间估算**：使用CuPy方案，整个项目可以在8-10周内完成，比纯CUDA C++方案节省约30-40%的开发时间。
