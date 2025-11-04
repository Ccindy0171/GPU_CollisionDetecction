# GPU碰撞检测算法设计文档

## 1. 整体介绍

本项目实现一种基于 GPU 的最近邻查找算法，完成快速的大规模碰撞检测。该算法可扩展至3D高斯溅射（3D Gaussian Splatting）场景中的应用。

### 1.1 项目目标
1. **GPU最近邻查找算法**：实现高效的并行碰撞检测
2. **性能测试与分析**：评估算法在不同规模下的性能表现
3. **刚体物理仿真**：模拟大量小球/物体的碰撞与运动
4. **3D Gaussians场景集成**：实现Gaussian场景与刚体的碰撞检测

### 1.2 算法框架
选择 **OpenGL + CUDA/Compute Shader** 作为实现框架，原因如下：
- 优秀的跨硬件平台支持
- 与3D Gaussian Splatting开源项目（https://github.com/graphdeco-inria/gaussian-splatting）技术栈一致
- 强大的并行计算能力
- 便于图形渲染与计算的集成

---

## 2. 核心算法设计

### 2.1 空间分割数据结构

#### 2.1.1 Uniform Grid（均匀网格）
**原理**：将3D空间划分为规则的立方体网格单元

**数据结构**：
```cpp
GridCell {
    int objectCount;        // 单元中的物体数量
    int startIndex;         // 在物体列表中的起始索引
}

GridConfig {
    vec3 worldMin;          // 世界空间最小坐标
    vec3 worldMax;          // 世界空间最大坐标
    ivec3 gridResolution;   // 网格分辨率（x, y, z维度的单元数）
    float cellSize;         // 单元大小
}
```

**GPU实现步骤**：
1. **清空网格**（GPU Kernel 1）：并行清空所有网格单元
2. **计算物体所在网格**（GPU Kernel 2）：每个线程处理一个物体，计算其网格坐标
3. **原子计数**（GPU Kernel 3）：使用原子操作累计每个网格的物体数量
4. **前缀和扫描**（GPU Kernel 4）：计算每个网格单元的起始索引
5. **填充网格**（GPU Kernel 5）：将物体索引写入对应网格

**优点**：
- 实现简单，GPU友好
- 查询速度快（O(1)定位邻居单元）
- 内存访问模式规则

**缺点**：
- 物体分布不均时效率低
- 需要预先知道场景范围

#### 2.1.2 Z-Order Curve（Morton Code）优化
使用Morton编码对网格单元进行空间排序，提高缓存局部性：

```cpp
uint mortonEncode(ivec3 coord) {
    // 将3D坐标交错位编码为1D Morton码
    uint x = expandBits(coord.x);
    uint y = expandBits(coord.y);
    uint z = expandBits(coord.z);
    return x | (y << 1) | (z << 2);
}
```

### 2.2 最近邻查找算法

#### 2.2.1 Broad Phase（粗检测阶段）
使用Uniform Grid快速筛选潜在碰撞对：

```glsl
// Compute Shader伪代码
void broadPhaseDetection(uint objectID) {
    Object obj = objects[objectID];
    ivec3 gridPos = worldToGrid(obj.position);
    
    // 检查周围27个网格单元（3x3x3）
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            for(int dz = -1; dz <= 1; dz++) {
                ivec3 neighborPos = gridPos + ivec3(dx, dy, dz);
                if(!isValidGrid(neighborPos)) continue;
                
                uint cellIndex = gridToIndex(neighborPos);
                GridCell cell = grid[cellIndex];
                
                // 遍历该单元中的所有物体
                for(int i = 0; i < cell.objectCount; i++) {
                    uint otherID = cellObjects[cell.startIndex + i];
                    if(otherID <= objectID) continue; // 避免重复检测
                    
                    // 添加到潜在碰撞对列表
                    addPotentialPair(objectID, otherID);
                }
            }
        }
    }
}
```

#### 2.2.2 Narrow Phase（精确检测阶段）
对粗检测筛选出的碰撞对进行精确检测：

```glsl
void narrowPhaseDetection(CollisionPair pair) {
    Object objA = objects[pair.idA];
    Object objB = objects[pair.idB];
    
    vec3 delta = objB.position - objA.position;
    float distance = length(delta);
    float minDist = objA.radius + objB.radius;
    
    if(distance < minDist) {
        // 发生碰撞，记录碰撞信息
        Collision collision;
        collision.normal = normalize(delta);
        collision.penetration = minDist - distance;
        collision.contactPoint = objA.position + collision.normal * objA.radius;
        
        collisions[atomicAdd(collisionCount, 1)] = collision;
    }
}
```

### 2.3 并行碰撞检测流程

**Pipeline设计**：
```
Frame N:
  1. Update Grid Structure (GPU)
     └─> 重新构建空间网格
  
  2. Broad Phase Detection (GPU)
     └─> 生成潜在碰撞对列表
  
  3. Narrow Phase Detection (GPU)
     └─> 精确碰撞检测
  
  4. Collision Response (GPU)
     └─> 计算碰撞响应
  
  5. Physics Integration (GPU)
     └─> 更新速度和位置
  
  6. Render (GPU)
     └─> 绘制场景
```

---

## 3. 物理仿真系统

### 3.1 刚体动力学

#### 3.1.1 物体属性
```cpp
struct RigidBody {
    vec3 position;          // 位置
    vec3 velocity;          // 速度
    vec3 acceleration;      // 加速度
    
    float radius;           // 半径（球体）
    float mass;             // 质量
    float restitution;      // 弹性系数 [0, 1]
    float friction;         // 摩擦系数
    
    vec3 color;             // 渲染颜色
    uint gridIndex;         // 当前所在网格
};
```

#### 3.1.2 碰撞响应
使用冲量法（Impulse-Based Method）：

```glsl
void resolveCollision(inout RigidBody objA, inout RigidBody objB, Collision col) {
    vec3 relativeVelocity = objB.velocity - objA.velocity;
    float velocityAlongNormal = dot(relativeVelocity, col.normal);
    
    // 物体正在分离，不处理
    if(velocityAlongNormal > 0) return;
    
    // 综合弹性系数
    float e = min(objA.restitution, objB.restitution);
    
    // 计算冲量标量
    float j = -(1.0 + e) * velocityAlongNormal;
    j /= (1.0 / objA.mass + 1.0 / objB.mass);
    
    // 应用冲量
    vec3 impulse = j * col.normal;
    objA.velocity -= impulse / objA.mass;
    objB.velocity += impulse / objB.mass;
    
    // 位置修正（避免穿透）
    float percent = 0.8; // 修正百分比
    float slop = 0.01;   // 允许的穿透量
    vec3 correction = max(col.penetration - slop, 0.0) / 
                      (1.0/objA.mass + 1.0/objB.mass) * percent * col.normal;
    objA.position -= correction / objA.mass;
    objB.position += correction / objB.mass;
}
```

#### 3.1.3 边界碰撞
```glsl
void handleBoundaryCollision(inout RigidBody obj, vec3 boundMin, vec3 boundMax) {
    for(int axis = 0; axis < 3; axis++) {
        // 下边界
        if(obj.position[axis] - obj.radius < boundMin[axis]) {
            obj.position[axis] = boundMin[axis] + obj.radius;
            obj.velocity[axis] = -obj.velocity[axis] * obj.restitution;
        }
        // 上边界
        if(obj.position[axis] + obj.radius > boundMax[axis]) {
            obj.position[axis] = boundMax[axis] - obj.radius;
            obj.velocity[axis] = -obj.velocity[axis] * obj.restitution;
        }
    }
}
```

#### 3.1.4 时间积分
使用Verlet积分或半隐式Euler方法：

```glsl
void integratePhysics(inout RigidBody obj, float dt) {
    // 半隐式Euler（更稳定）
    vec3 acceleration = obj.acceleration + gravity;
    obj.velocity += acceleration * dt;
    obj.position += obj.velocity * dt;
    
    // 应用阻尼（空气阻力）
    obj.velocity *= (1.0 - damping * dt);
}
```

### 3.2 场景设置

#### 3.2.1 初始化策略
```cpp
// 随机生成策略
void generateRandomSpheres(int count) {
    for(int i = 0; i < count; i++) {
        RigidBody sphere;
        sphere.position = randomInBox(sceneMin, sceneMax);
        sphere.velocity = randomVelocity(-10, 10);
        sphere.radius = randomFloat(0.1, 0.5);
        sphere.mass = sphere.radius * sphere.radius * sphere.radius * density;
        sphere.restitution = randomFloat(0.5, 0.95);
        spheres.push_back(sphere);
    }
}

// 预设场景策略
void createPresetScene() {
    // 示例：重力下落场景
    createSphereGrid(10, 10, 10, spacing=1.0, height=20.0);
    
    // 示例：爆炸扩散场景
    createSphereCluster(center, count=1000, explosionForce=50.0);
}
```

---

## 4. 3D Gaussians场景碰撞检测

### 4.1 3D Gaussian Splatting简介
3D Gaussians使用各向异性高斯分布表示场景：

```cpp
struct Gaussian {
    vec3 position;          // 中心位置 μ
    vec3 scale;             // 缩放参数
    vec4 rotation;          // 四元数表示旋转
    vec3 color;             // 颜色（SH系数）
    float opacity;          // 不透明度
};
```

每个Gaussian定义一个3D椭球，可用协方差矩阵表示：
```
Σ = R S S^T R^T
```
其中R是旋转矩阵，S是缩放矩阵。

### 4.2 Gaussian-Sphere碰撞检测

#### 4.2.1 椭球-球体相交测试
```glsl
bool gaussianSphereIntersection(Gaussian g, RigidBody sphere, out vec3 contactNormal) {
    // 1. 将球心变换到Gaussian局部坐标系
    mat3 R = quaternionToMatrix(g.rotation);
    vec3 localPos = inverse(R) * (sphere.position - g.position);
    
    // 2. 在局部空间进行椭球-球体测试
    vec3 scaledPos = localPos / g.scale;
    float distSquared = dot(scaledPos, scaledPos);
    
    // 3. 判断球心到椭球中心的"标准化距离"
    float threshold = 1.0 + (sphere.radius / length(g.scale));
    
    if(distSquared < threshold * threshold) {
        // 计算接触法线（在世界空间）
        vec3 localNormal = normalize(scaledPos / g.scale);
        contactNormal = R * localNormal;
        return true;
    }
    
    return false;
}
```

#### 4.2.2 基于不透明度的碰撞
```glsl
float gaussianDensity(Gaussian g, vec3 worldPos) {
    mat3 R = quaternionToMatrix(g.rotation);
    vec3 localPos = inverse(R) * (worldPos - g.position);
    vec3 scaledPos = localPos / g.scale;
    
    float exponent = -0.5 * dot(scaledPos, scaledPos);
    return g.opacity * exp(exponent);
}

bool checkGaussianCollision(vec3 spherePos, float threshold) {
    float totalDensity = 0.0;
    
    // 累积附近Gaussians的密度
    for(int i = 0; i < nearbyGaussianCount; i++) {
        totalDensity += gaussianDensity(nearbyGaussians[i], spherePos);
    }
    
    return totalDensity > threshold;
}
```

### 4.3 加速结构

#### 4.3.1 Gaussian空间索引
```cpp
// 为Gaussians建立Uniform Grid
struct GaussianGrid {
    vector<vector<int>> cells;  // 每个单元包含的Gaussian索引
    ivec3 resolution;
    vec3 cellSize;
};

void buildGaussianGrid(const vector<Gaussian>& gaussians) {
    for(int i = 0; i < gaussians.size(); i++) {
        // 计算Gaussian包围盒
        AABB bbox = computeGaussianAABB(gaussians[i]);
        
        // 将Gaussian添加到所有相交的单元
        ivec3 minCell = worldToGrid(bbox.min);
        ivec3 maxCell = worldToGrid(bbox.max);
        
        for(int x = minCell.x; x <= maxCell.x; x++) {
            for(int y = minCell.y; y <= maxCell.y; y++) {
                for(int z = minCell.z; z <= maxCell.z; z++) {
                    int cellIdx = gridIndex(x, y, z);
                    gaussianGrid.cells[cellIdx].push_back(i);
                }
            }
        }
    }
}
```

#### 4.3.2 分层碰撞检测
```glsl
void detectGaussianCollisions(inout RigidBody sphere) {
    ivec3 gridPos = worldToGrid(sphere.position);
    
    vec3 totalForce = vec3(0.0);
    int collisionCount = 0;
    
    // 检查周围单元中的Gaussians
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            for(int dz = -1; dz <= 1; dz++) {
                ivec3 cellPos = gridPos + ivec3(dx, dy, dz);
                int cellIdx = gridIndex(cellPos);
                
                for(int i = 0; i < gaussianGrid[cellIdx].count; i++) {
                    Gaussian g = gaussians[gaussianGrid[cellIdx].indices[i]];
                    
                    vec3 contactNormal;
                    if(gaussianSphereIntersection(g, sphere, contactNormal)) {
                        // 计算斥力
                        float penetration = computePenetration(g, sphere);
                        vec3 force = contactNormal * penetration * stiffness;
                        totalForce += force;
                        collisionCount++;
                    }
                }
            }
        }
    }
    
    if(collisionCount > 0) {
        // 应用平均力
        vec3 avgForce = totalForce / float(collisionCount);
        sphere.velocity += avgForce / sphere.mass * dt;
    }
}
```

---

## 5. 性能优化策略

### 5.1 GPU并行优化

#### 5.1.1 Work Group优化
```glsl
// 推荐配置
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// 使用共享内存减少全局内存访问
shared uint sharedCollisionCount[256];
```

#### 5.1.2 内存合并访问
```cpp
// 按照SOA（Structure of Arrays）组织数据
struct PhysicsData {
    float* positions_x;    // 所有x坐标
    float* positions_y;
    float* positions_z;
    float* velocities_x;
    // ...
};
```

#### 5.1.3 Atomic操作优化
```glsl
// 使用local atomic减少全局atomic操作
shared uint localCounter;

if(localID == 0) localCounter = 0;
barrier();

atomicAdd(localCounter, 1);
barrier();

if(localID == 0) {
    atomicAdd(globalCounter, localCounter);
}
```

### 5.2 算法优化

#### 5.2.1 动态网格调整
```cpp
void updateGridResolution(const vector<RigidBody>& objects) {
    // 根据物体密度动态调整网格大小
    float avgRadius = computeAverageRadius(objects);
    float optimalCellSize = 2.0 * avgRadius;
    gridConfig.cellSize = optimalCellSize;
    gridConfig.resolution = computeResolution(worldSize, optimalCellSize);
}
```

#### 5.2.2 空间剔除
```glsl
// 视锥体剔除
bool isInFrustum(vec3 position, float radius) {
    for(int i = 0; i < 6; i++) {
        if(dot(frustumPlanes[i], vec4(position, 1.0)) < -radius) {
            return false;
        }
    }
    return true;
}
```

#### 5.2.3 休眠机制
```cpp
// 静止物体休眠
if(length(obj.velocity) < sleepThreshold && 
   length(obj.acceleration) < sleepThreshold) {
    obj.isSleeping = true;
    // 跳过物理更新
}
```

---

## 6. 性能测试方案

### 6.1 测试指标

#### 6.1.1 性能指标
- **FPS（帧率）**：实时渲染性能
- **碰撞检测时间**：每帧碰撞检测的GPU时间
- **网格构建时间**：空间数据结构构建时间
- **物理更新时间**：物理仿真计算时间
- **内存使用**：GPU显存占用

#### 6.1.2 准确性指标
- **漏检率**：未检测到的实际碰撞比例
- **误检率**：错误检测的碰撞比例
- **穿透深度误差**：碰撞响应的精度

### 6.2 测试场景

```cpp
TestScenario scenarios[] = {
    // 场景1：规模测试
    {"Scale_1K",   1000,   uniformDistribution},
    {"Scale_5K",   5000,   uniformDistribution},
    {"Scale_10K",  10000,  uniformDistribution},
    {"Scale_50K",  50000,  uniformDistribution},
    {"Scale_100K", 100000, uniformDistribution},
    
    // 场景2：密度测试
    {"Density_Low",    10000, sparseDistribution},
    {"Density_Medium", 10000, mediumDistribution},
    {"Density_High",   10000, denseDistribution},
    
    // 场景3：动态场景
    {"Dynamic_Explosion", 5000, explosionPattern},
    {"Dynamic_Fountain",  5000, fountainPattern},
    {"Dynamic_Avalanche", 5000, avalanchePattern},
};
```

### 6.3 性能分析工具

```cpp
class PerformanceProfiler {
public:
    void beginFrame();
    void endFrame();
    
    void beginEvent(const char* name);
    void endEvent();
    
    void recordMetric(const char* name, float value);
    void exportResults(const char* filename);
    
private:
    GLuint queryObjects[MAX_QUERIES];
    std::map<std::string, std::vector<float>> metrics;
};

// 使用示例
profiler.beginEvent("BroadPhase");
broadPhaseDetection();
profiler.endEvent();
```

---

## 7. 实现路线图

### Phase 1: 基础框架（Week 1-2）
- [ ] OpenGL/CUDA环境搭建
- [ ] 基础渲染管线
- [ ] 简单的CPU碰撞检测原型
- [ ] 基础物理引擎（Euler积分）

### Phase 2: GPU空间数据结构（Week 3-4）
- [ ] Uniform Grid的GPU实现
- [ ] Compute Shader编写
- [ ] 内存管理与数据传输优化
- [ ] Morton Code优化

### Phase 3: 碰撞检测与响应（Week 5-6）
- [ ] Broad Phase GPU实现
- [ ] Narrow Phase GPU实现
- [ ] 碰撞响应（冲量法）
- [ ] 边界处理

### Phase 4: 场景仿真（Week 7-8）
- [ ] 多样化初始化场景
- [ ] 交互控制（鼠标、键盘）
- [ ] 相机系统
- [ ] 动画录制功能

### Phase 5: 性能测试（Week 9）
- [ ] 性能测试框架
- [ ] 多场景测试
- [ ] 数据收集与可视化
- [ ] 性能报告撰写

### Phase 6: 3D Gaussians集成（Week 10-12）
- [ ] Gaussian数据加载
- [ ] Gaussian-Sphere碰撞检测
- [ ] Gaussian渲染集成
- [ ] 混合场景仿真

### Phase 7: 优化与完善（Week 13-14）
- [ ] 性能优化
- [ ] Bug修复
- [ ] 文档完善
- [ ] 最终演示视频制作

---

## 8. 技术栈总结

### 8.1 开发工具
- **语言**：C++17, GLSL 4.5+, CUDA (可选)
- **图形API**：OpenGL 4.5+
- **库依赖**：
  - GLM（数学库）
  - GLFW（窗口管理）
  - GLAD（OpenGL加载器）
  - ImGui（UI界面）
  - stb_image（纹理加载）

### 8.2 可选扩展
- **CUDA**：更高性能的GPU计算
- **Vulkan Compute**：跨平台计算着色器
- **OptiX**：光线追踪加速
- **3D Gaussian Splatting原始实现**：集成已有渲染器

---

## 9. 预期成果

### 9.1 技术成果
1. **高效GPU碰撞检测系统**
   - 支持10万+物体的实时碰撞检测
   - 帧率 > 30 FPS（取决于硬件）

2. **完整物理仿真引擎**
   - 刚体动力学
   - 弹性碰撞
   - 边界交互

3. **3D Gaussians场景集成**
   - Gaussian场景渲染
   - Gaussian-刚体碰撞
   - 混合渲染管线

### 9.2 可视化成果
1. **基础仿真动画**（30-60秒）
   - 重力下落场景
   - 爆炸扩散场景
   - 边界弹跳场景

2. **Gaussian场景动画**（30-60秒）
   - 小球在Gaussian场景中的运动
   - Gaussian碰撞效果展示

3. **性能分析报告**
   - 性能曲线图
   - 对比实验数据
   - 优化效果展示

---

## 10. 参考文献

1. **空间数据结构**
   - "Real-Time Collision Detection" - Christer Ericson
   - "Foundations of Physically Based Modeling and Animation" - Donald House

2. **GPU并行算法**
   - "GPU Gems 3: Chapter 32. Broad-Phase Collision Detection with CUDA"
   - "Parallel Algorithms for Collision Detection"

3. **3D Gaussian Splatting**
   - "3D Gaussian Splatting for Real-Time Radiance Field Rendering" - Kerbl et al., 2023
   - Original implementation: https://github.com/graphdeco-inria/gaussian-splatting

4. **物理仿真**
   - "Game Physics Engine Development" - Ian Millington
   - "Real-Time Rendering, 4th Edition" - Tomas Akenine-Möller et al.

---

## 附录：伪代码总览

```cpp
// 主循环
void mainLoop() {
    while(!shouldClose) {
        // 1. 输入处理
        handleInput();
        
        // 2. 更新空间网格（GPU）
        updateUniformGrid(rigidBodies);
        
        // 3. 碰撞检测（GPU）
        auto pairs = broadPhaseDetection(rigidBodies, grid);
        auto collisions = narrowPhaseDetection(pairs);
        
        // 4. 碰撞响应（GPU）
        resolveCollisions(collisions, rigidBodies);
        
        // 5. 物理积分（GPU）
        integratePhysics(rigidBodies, dt);
        
        // 6. Gaussian碰撞（可选，GPU）
        if(gaussianSceneEnabled) {
            detectGaussianCollisions(rigidBodies, gaussians);
        }
        
        // 7. 渲染（GPU）
        render(rigidBodies, gaussians);
        
        // 8. 性能统计
        profiler.recordFrame();
        
        swapBuffers();
    }
}
```