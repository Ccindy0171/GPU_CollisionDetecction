# System Architecture

This document describes the architecture, module relationships, and program flow of the GPU Collision Detection system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Relationships](#module-relationships)
3. [Data Flow](#data-flow)
4. [Main Program Flow](#main-program-flow)
5. [Key Algorithms](#key-algorithms)

## System Overview

The GPU Collision Detection system is a high-performance physics simulation framework that leverages GPU parallel computing for real-time collision detection and response. The system is designed with a modular architecture that separates concerns:

- **Physics Engine**: Handles rigid body dynamics and collision response
- **Spatial Acceleration**: Uses uniform grid for O(N) collision detection
- **Visualization**: OpenGL-based 3D rendering with realistic lighting
- **Performance Monitoring**: Tracks and analyzes performance metrics

### Design Principles

1. **GPU-First**: All compute-intensive operations run on GPU
2. **Data Locality**: Structure of Arrays (SOA) layout for coalesced memory access
3. **Minimal CPU-GPU Transfer**: Data stays on GPU as much as possible
4. **Modular Design**: Clear separation between physics, rendering, and utilities

## Module Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ examples/        │  │ tests/           │                │
│  │ - gravity_fall   │  │ - unit tests     │                │
│  │ - demos          │  │ - benchmarks     │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─── uses ───┐
                            │             │
┌─────────────────────────────────────────────────────────────┐
│                      Core Library (src/)                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              PhysicsSimulator (simulator.py)         │  │
│  │  - Main simulation loop                              │  │
│  │  - Orchestrates all components                       │  │
│  └──────────────────────────────────────────────────────┘  │
│           │           │            │           │            │
│           ├───────────┼────────────┼───────────┤            │
│           │           │            │           │            │
│  ┌────────▼────┐ ┌───▼─────┐ ┌───▼────┐ ┌───▼────────┐   │
│  │ RigidBody   │ │ Spatial │ │ CUDA   │ │ Performance│   │
│  │ System      │ │ Grid    │ │ Kernels│ │ Monitor    │   │
│  │             │ │         │ │        │ │            │   │
│  │ rigid_body  │ │ spatial │ │kernels │ │performance │   │
│  │ .py         │ │ _grid.py│ │.py     │ │.py         │   │
│  │             │ │         │ │        │ │            │   │
│  │ • positions │ │• uniform│ │• hash  │ │• timing    │   │
│  │ • velocities│ │  grid   │ │• detect│ │• metrics   │   │
│  │ • forces    │ │• spatial│ │• resolve│ │• plotting  │   │
│  │ • properties│ │  hash   │ │• integr│ │            │   │
│  └─────────────┘ └─────────┘ └────────┘ └────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Visualization (opengl_visualizer.py)          │  │
│  │  - OpenGL rendering                                  │  │
│  │  - Camera controls                                   │  │
│  │  - Video recording                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Utilities (init_helper.py)                 │  │
│  │  - Initialization helpers                            │  │
│  │  - Object placement                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ depends on
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  External Dependencies                       │
│  • CuPy (GPU computing)                                     │
│  • NumPy (numerical arrays)                                 │
│  • PyOpenGL (3D rendering)                                  │
│  • OpenCV (video recording)                                 │
└─────────────────────────────────────────────────────────────┘
```

### Module Descriptions

#### 1. **simulator.py** (PhysicsSimulator)
   - **Role**: Main orchestrator of the simulation
   - **Dependencies**: rigid_body, spatial_grid, kernels, performance
   - **Responsibilities**:
     - Initialize physics system
     - Execute simulation loop
     - Coordinate collision detection pipeline
     - Manage performance metrics

#### 2. **rigid_body.py** (RigidBodySystem)
   - **Role**: Manages all physics objects and their properties
   - **Dependencies**: CuPy, NumPy
   - **Data Structure**: Structure of Arrays (SOA) for GPU efficiency
   - **Responsibilities**:
     - Store object positions, velocities, forces
     - Store physical properties (mass, radius, restitution)
     - Provide CPU↔GPU data transfer methods
     - Calculate system-wide properties (kinetic energy, momentum)

#### 3. **spatial_grid.py** (UniformGrid)
   - **Role**: Spatial acceleration structure for fast neighbor queries
   - **Dependencies**: CuPy, NumPy
   - **Algorithm**: Uniform spatial hashing
   - **Responsibilities**:
     - Partition 3D space into regular grid cells
     - Map positions to grid cells (spatial hashing)
     - Store object-to-cell mappings
     - Provide neighbor cell queries

#### 4. **kernels.py** (CUDA Kernels)
   - **Role**: GPU-accelerated compute kernels
   - **Dependencies**: CuPy RawKernel
   - **Language**: CUDA C (compiled at runtime)
   - **Kernels**:
     - `COMPUTE_GRID_HASH_KERNEL`: Position → grid hash mapping
     - `FIND_CELL_START_KERNEL`: Build cell start/end indices
     - `BROAD_PHASE_KERNEL`: Detect collision pairs
     - `COLLISION_RESPONSE_KERNEL`: Resolve collisions (impulse method)
     - `INTEGRATE_KERNEL`: Physics integration (semi-implicit Euler)

#### 5. **opengl_visualizer.py** (OpenGLVisualizer)
   - **Role**: 3D visualization and rendering
   - **Dependencies**: PyOpenGL, GLUT, GLU
   - **Features**:
     - Phong lighting model
     - Camera controls (rotate, zoom, pan)
     - Real-time rendering
     - Video recording (MP4 export)

#### 6. **performance.py** (PerformanceMonitor)
   - **Role**: Performance tracking and analysis
   - **Dependencies**: CuPy (for GPU timing), matplotlib
   - **Capabilities**:
     - GPU/CPU event timing
     - Metric collection and statistics
     - Performance visualization (plots, charts)
     - CSV export

#### 7. **init_helper.py** (Utility Functions)
   - **Role**: Initialization and setup utilities
   - **Dependencies**: NumPy
   - **Functions**:
     - Generate non-overlapping object positions
     - Verify initialization constraints
     - Scene setup helpers

## Data Flow

### Initialization Phase

```
User Script
    │
    ├─> Create PhysicsSimulator
    │       │
    │       ├─> Initialize RigidBodySystem (on GPU)
    │       ├─> Initialize UniformGrid (on GPU)
    │       └─> Compile CUDA kernels
    │
    ├─> Set initial positions (CPU → GPU)
    ├─> Set velocities, masses, radii (CPU → GPU)
    └─> (Optional) Create OpenGLVisualizer
```

### Simulation Loop (per frame)

```
PhysicsSimulator.step()
    │
    ├─> 1. INTEGRATE (Update positions/velocities)
    │   │   • Apply forces (gravity, external)
    │   │   • Semi-implicit Euler integration
    │   │   • Handle boundary collisions
    │   │   [CUDA Kernel: INTEGRATE_KERNEL]
    │   └─> Updated positions, velocities (on GPU)
    │
    ├─> 2. BUILD_GRID (Spatial partitioning)
    │   │   • Compute grid hash for each object
    │   │   • Sort objects by hash
    │   │   • Find cell start/end indices
    │   │   [CUDA Kernels: COMPUTE_GRID_HASH, FIND_CELL_START]
    │   └─> Grid data structure (on GPU)
    │
    ├─> 3. DETECT_COLLISIONS (Broad phase)
    │   │   • Check 27 neighbor cells per object
    │   │   • Sphere-sphere intersection tests
    │   │   • Build collision pair list
    │   │   [CUDA Kernel: BROAD_PHASE_KERNEL]
    │   └─> Collision pairs array (on GPU)
    │
    ├─> 4. RESOLVE_COLLISIONS (Collision response)
    │   │   • Calculate collision normal
    │   │   • Apply impulse (velocity change)
    │   │   • Position correction (separation)
    │   │   [CUDA Kernel: COLLISION_RESPONSE_KERNEL]
    │   └─> Updated velocities, positions (on GPU)
    │
    ├─> 5. (Optional) VISUALIZE
    │   │   • Copy positions, colors to CPU
    │   │   • Render with OpenGL
    │   └─> Display frame / Save to video
    │
    └─> Return statistics
```

### Memory Layout

All arrays use **Structure of Arrays (SOA)** for GPU efficiency:

```
Positions:  [x₁, y₁, z₁, x₂, y₂, z₂, ..., xₙ, yₙ, zₙ]  # shape: (N, 3)
Velocities: [vx₁, vy₁, vz₁, vx₂, vy₂, vz₂, ..., vxₙ, vyₙ, vzₙ]
Radii:      [r₁, r₂, r₃, ..., rₙ]  # shape: (N,)
Masses:     [m₁, m₂, m₃, ..., mₙ]
```

Benefits:
- Coalesced memory access (threads read consecutive addresses)
- Better cache utilization
- Efficient vectorization

## Main Program Flow

### High-Level Flow

```python
# 1. Initialization
simulator = PhysicsSimulator(
    num_objects=1000,
    world_bounds=((-50, 0, -50), (50, 50, 50)),
    cell_size=2.0,
    dt=1/60
)

# Set initial state (on GPU)
simulator.bodies.set_positions(initial_positions)
simulator.bodies.set_velocities(initial_velocities)

# 2. Simulation Loop
for frame in range(num_frames):
    # Execute one physics step (all on GPU)
    stats = simulator.step()
    
    # (Optional) Visualize
    if visualizer:
        visualizer.render(
            simulator.bodies.positions.get(),  # GPU → CPU
            simulator.bodies.radii.get()
        )

# 3. Analysis
print(f"Avg FPS: {1000 / stats['total_time']:.1f}")
```

### Detailed Step-by-Step Flow

#### Step 1: Physics Integration

```
For each object (in parallel on GPU):
    1. Read: position, velocity, force, mass, radius
    2. Compute: acceleration = force/mass + gravity
    3. Update: velocity += acceleration * dt  (semi-implicit)
    4. Update: position += velocity * dt
    5. Apply: velocity damping (air resistance)
    6. Check: boundary collisions
        If hit boundary:
            - Clamp position to boundary
            - Reverse velocity component
            - Apply restitution coefficient
    7. Write: updated position, velocity
```

#### Step 2: Grid Construction

```
Phase A - Compute Hashes:
    For each object (in parallel):
        1. grid_coord = floor((position - world_min) / cell_size)
        2. hash = z * (res_y * res_x) + y * res_x + x
        3. Store hash

Phase B - Sort:
    Sort object indices by hash value (GPU parallel sort)

Phase C - Find Cell Boundaries:
    For each object (in parallel):
        If first in cell: mark cell_start[hash] = index
        If last in cell: mark cell_end[hash] = index + 1
```

#### Step 3: Collision Detection (Broad Phase)

```
For each object i (in parallel):
    1. Determine object i's grid cell
    2. For each of 27 neighbor cells:
        3. Get objects in that cell (via cell_start/end)
        4. For each object j in cell:
            If i < j:  # Avoid duplicates
                5. dist = ||pos_i - pos_j||
                6. If dist < radius_i + radius_j:
                    7. Atomically add pair (i,j) to collision list
```

#### Step 4: Collision Response

```
For each collision pair (i,j) (in parallel):
    1. Calculate collision normal: n = (pos_j - pos_i) / ||pos_j - pos_i||
    2. Calculate relative velocity: v_rel = vel_j - vel_i
    3. Calculate velocity along normal: v_n = v_rel · n
    
    If v_n < 0 (approaching):
        4. Calculate impulse: j = -(1+e) * v_n / (1/m_i + 1/m_j)
        5. Apply impulse to velocities:
            vel_i -= j * n / m_i  (atomic)
            vel_j += j * n / m_j  (atomic)
    
    If penetrating:
        6. Calculate penetration depth: p = (r_i + r_j) - dist
        7. Separate objects proportional to mass:
            pos_i -= correction_i * n  (atomic)
            pos_j += correction_j * n  (atomic)
```

## Key Algorithms

### 1. Uniform Grid Spatial Hashing

**Purpose**: Reduce collision detection from O(N²) to O(N·k) where k = avg objects per cell

**Algorithm**:
```
1. Partition space into regular grid cells
2. Assign each object to a cell based on position
3. For collision queries, only check objects in nearby cells (3×3×3)
```

**Cell Size Selection**: 
- Optimal: 2× average object diameter
- Too small: excessive memory, many cells to check
- Too large: many objects per cell, slower collision checks

### 2. Impulse-Based Collision Response

**Purpose**: Physically accurate collision resolution

**Algorithm**:
```
1. Calculate collision normal (direction)
2. Calculate relative velocity along normal
3. Compute impulse magnitude: j = -(1+e) * v_n / (1/m_a + 1/m_b)
4. Apply impulse: Δv = j/m
5. Position correction to resolve penetration
```

**Coefficient of Restitution (e)**:
- e=0: Perfectly inelastic (objects stick)
- e=1: Perfectly elastic (perfect bounce)
- Real materials: e ∈ (0.3, 0.95)

### 3. Semi-Implicit Euler Integration

**Purpose**: Stable, energy-conserving time integration

**Algorithm**:
```
1. v(t+Δt) = v(t) + a(t) * Δt     # Update velocity first
2. x(t+Δt) = x(t) + v(t+Δt) * Δt  # Use new velocity for position
```

**Advantages over explicit Euler**:
- More stable for large timesteps
- Better energy conservation
- Prevents "explosion" artifacts

## Performance Considerations

### GPU Optimization Techniques

1. **Coalesced Memory Access**:
   - SOA layout ensures adjacent threads access adjacent memory
   - Maximizes memory bandwidth utilization

2. **Minimized Atomic Operations**:
   - Atomic ops are slow (serialized)
   - Used only when necessary (collision pair counting, velocity updates)
   - Mitigated by having many collision pairs processed in parallel

3. **Optimal Block Size**:
   - 256 threads per block
   - Good occupancy on most NVIDIA GPUs
   - Balances register usage and parallelism

4. **Minimal CPU-GPU Transfer**:
   - Data stays on GPU across frames
   - Only transfer to CPU for visualization/analysis
   - Use pinned memory for faster transfers when needed

### Scalability

| Objects | Grid Cells | Avg Objects/Cell | Complexity |
|---------|------------|------------------|------------|
| 1,000   | ~10,000    | 0.1             | ~270 checks/object |
| 10,000  | ~10,000    | 1.0             | ~270 checks/object |
| 100,000 | ~100,000   | 1.0             | ~270 checks/object |

The uniform grid keeps complexity linear O(N) as long as objects are reasonably distributed.

## References

- **Collision Detection**: "Real-Time Collision Detection" by Christer Ericson
- **GPU Algorithms**: "GPU Gems 3, Chapter 32: Broad-Phase Collision Detection with CUDA"
- **Physics Simulation**: "Game Physics Engine Development" by Ian Millington
- **Numerical Integration**: "Foundations of Physically Based Modeling" by Witkin & Baraff
