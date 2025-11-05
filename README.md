````markdown
# GPU Collision Detection System

A high-performance GPU-accelerated collision detection and physics simulation system using CuPy and OpenGL.

## Project Overview

This project implements a complete GPU-accelerated collision detection system with:

1. **Uniform Grid-Based Spatial Search** - O(N) complexity for large-scale collision detection
2. **Complete Physics Engine** - Rigid body dynamics, collision response, and boundary handling
3. **High-Quality 3D Visualization** - Real-time OpenGL rendering with Phong shading
4. **High-Definition Video Export** - H.264 encoded MP4 output

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Tested on: RTX 3050 (4GB VRAM)
- Recommended: 8GB+ system RAM

### Software
- Python 3.8+
- CUDA 11.x or 12.x
- Linux / Windows / macOS

## Installation

### 1. Install CUDA

Verify NVIDIA driver and CUDA Toolkit installation:

```bash
nvcc --version
```

### 2. Create Virtual Environment (Recommended)

```bash
cd GPU_CollisionDetecction
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: Choose the correct CuPy package for your CUDA version:
- CUDA 12.x: `pip install cupy-cuda12x` (recommended)
- CUDA 11.x: `pip install cupy-cuda11x`

### 4. Verify Installation

```bash
python -c "import cupy as cp; print('CuPy:', cp.__version__); print('CUDA:', cp.cuda.is_available())"
python -c "import OpenGL; print('OpenGL available')"
```

## Project Structure

```
GPU_CollisionDetecction/
├── src/                          # Core implementation
│   ├── __init__.py              # Public API exports
│   ├── rigid_body.py            # Rigid body physics
│   ├── spatial_grid.py          # Uniform grid data structure
│   ├── kernels.py               # CUDA kernel functions
│   ├── simulator.py             # Main physics simulator
│   ├── opengl_visualizer.py     # OpenGL 3D renderer
│   ├── init_helper.py           # Initialization utilities
│   └── performance.py           # Performance monitoring
├── examples/
│   └── gravity_fall.py          # Gravity drop simulation (main example)
├── tests/
│   ├── test_01_head_on.py       # Head-on collision test
│   ├── test_02_static_overlap.py # Static overlap resolution test
│   ├── test_03_falling_balls.py  # Multiple falling balls test
│   ├── test_04_large_scale.py    # Large-scale performance test
│   ├── test_opengl_basic.py      # OpenGL functionality test
│   └── test_physics_only.py      # Physics validation (no rendering)
├── output/                       # Generated files (videos, logs)
├── algorithm_design.md           # Algorithm documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick Start

### Run the Main Example

Launch the gravity fall simulation with 500 spheres:

```bash
python examples/gravity_fall.py
```

**Output:**
- Real-time 3D visualization window
- `output/gravity_fall_opengl.mp4` - High-quality video (1920×1080, 60fps, H.264)

**Interactive Controls:**
- **Left Mouse**: Rotate camera
- **Right Mouse**: Zoom in/out
- **Middle Mouse**: Pan camera
- **Space**: Pause/Resume
- **G**: Toggle grid
- **A**: Toggle axes
- **W**: Toggle wireframe
- **R**: Reset camera
- **Q/ESC**: Quit

**Configuration** (edit in script):
```python
NUM_OBJECTS = 500           # Number of spheres
WORLD_BOUNDS = ((-20, 0, -20), (20, 40, 20))
CELL_SIZE = 2.0
NUM_FRAMES = 600            # 10 seconds @ 60fps
RECORD_VIDEO = True
```

### Run Tests

Test the physics system (no OpenGL display required):

```bash
# Test 1: Head-on collision between two spheres
python tests/test_01_head_on.py

# Test 2: Overlapping spheres resolution
python tests/test_02_static_overlap.py

# Test 3: Multiple falling spheres
python tests/test_03_falling_balls.py

# Test 4: Large-scale simulation (1000+ objects)
python tests/test_04_large_scale.py

# Physics validation (no rendering)
python tests/test_physics_only.py

# OpenGL functionality test
python tests/test_opengl_basic.py
```

## Core Algorithms

### Collision Detection Pipeline

```
1. Grid Construction
   ├─ Compute spatial hash for each object
   ├─ Sort by hash value
   └─ Find cell start/end positions

2. Broad Phase (Coarse Detection)
   └─ Check 27 neighboring cells (3×3×3 neighborhood)

3. Narrow Phase (Precise Detection)
   └─ Sphere-sphere intersection testing

4. Collision Response
   ├─ Impulse-based velocity correction
   └─ Position separation to prevent penetration

5. Physics Integration
   ├─ Semi-implicit Euler method
   └─ Boundary collision handling
```

### GPU Parallelization

All compute-intensive operations run in parallel on GPU:

- **Grid Hashing**: One thread per object
- **Collision Detection**: One thread per object checking neighbors
- **Collision Response**: One thread per collision pair
- **Integration**: One thread per object

## Performance Characteristics

### RTX 3050 Benchmark

| Objects | Frame Time | FPS | Avg Collisions |
|---------|------------|-----|-----------------|
| 500     | ~4ms       | 250 | ~300           |
| 1,000   | ~7ms       | 143 | ~600           |
| 5,000   | ~15ms      | 67  | ~1500          |
| 10,000  | ~25ms      | 40  | ~2500          |

**Optimal Configuration:**
- Real-time visualization: 500-2000 objects
- High-quality rendering: Up to 5000 objects
- Headless simulation: 10000+ objects

### Performance Tuning

1. **Adjust Grid Size**
   ```python
   CELL_SIZE = 2.0 * average_radius  # Optimal balance
   ```
   - Too small: Excessive memory usage
   - Too large: Many objects per cell, slower detection

2. **Reduce Rendering Overhead**
   - Use `test_physics_only.py` for headless benchmarking

3. **Control Object Count**
   - More objects = more collisions = slower simulation
   - Start with 500-1000 and adjust as needed

## Technical Details

### Memory Layout

Uses **SOA (Structure of Arrays)** for optimal GPU memory access:

```python
positions = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # [N, 3]
velocities = np.array([[vx1, vy1, vz1], ...])            # [N, 3]
```

Benefits:
- Coalesced memory access
- High cache hit rate
- Efficient vectorization

### CUDA Kernels

- **Block Size**: 256 threads (optimized for RTX 3050)
- **Memory Access**: Coalesced patterns
- **Atomic Operations**: Minimized
- **Registers**: Carefully tuned for occupancy

### OpenGL Rendering

- **Lighting**: Phong shading model
- **Anti-Aliasing**: MSAA 4x
- **Spheres**: GLU quadrics (32 slices/stacks)
- **Resolution**: 1920×1080 default

## Usage Examples

### Basic Simulation

```python
from src import PhysicsSimulator
import numpy as np

# Create simulator
sim = PhysicsSimulator(
    num_objects=1000,
    world_bounds=((-50, 0, -50), (50, 50, 50)),
    cell_size=2.0,
    gravity=(0, -9.81, 0),
    dt=1.0/60.0
)

# Set initial state
positions = np.random.uniform(-40, 40, (1000, 3)).astype(np.float32)
sim.bodies.positions[:] = positions

# Run simulation
for frame in range(600):
    stats = sim.step()
    print(f"Frame {frame}: {stats['num_collisions']} collisions")
```

### Visualize Results

```python
from src import OpenGLVisualizer
import numpy as np

vis = OpenGLVisualizer(
    world_bounds=((-50, 0, -50), (50, 50, 50)),
    width=1920,
    height=1080
)

# Custom render function
def render():
    positions = sim.bodies.positions.get()  # Copy from GPU
    radii = sim.bodies.radii.get()
    vis.render(positions, radii, colors)

vis.set_render_function(render)
vis.run()
```

## Troubleshooting

### Issue: "CUDARuntimeError: out of memory"

**Solution:**
- Reduce `NUM_OBJECTS`
- Increase `CELL_SIZE`
- Use headless mode (no visualization)

```python
# Reduce to fit your GPU
NUM_OBJECTS = 1000
CELL_SIZE = 2.5
```

### Issue: Low Frame Rate

**Causes & Solutions:**
1. Too many objects
   - Reduce `NUM_OBJECTS`
2. Grid too fine-grained
   - Increase `CELL_SIZE`
3. VRAM thrashing
   - Reduce resolution or object count

### Issue: CuPy Installation Fails

**Solution:**
```bash
# Verify CUDA version
nvcc --version

# Install matching CuPy
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x
```

### Issue: OpenGL Won't Display

**Solution:**
- Ensure X11 is running (Linux)
- Check graphics driver is up-to-date
- Try headless testing: `python tests/test_physics_only.py`

## Performance Optimization Tips

1. **Profile Your Code**
   ```python
   from src import PerformanceMonitor
   monitor = PerformanceMonitor()
   monitor.record_metric('kernel_time', time)
   monitor.print_statistics()
   ```

2. **Use Appropriate Grid Size**
   - Too small: 27+ cells checked per object
   - Too large: Dense cells with many collision checks
   - Sweet spot: 2-3 objects per cell average

3. **Batch Operations**
   - Process multiple frames before GPU→CPU transfer
   - Reduces bandwidth overhead

## Future Improvements

- [ ] Support for non-spherical shapes (boxes, capsules)
- [ ] BVH (Bounding Volume Hierarchy) acceleration
- [ ] Multi-GPU support
- [ ] Interactive real-time controls
- [ ] Soft body physics
- [ ] Cloth simulation
- [ ] Fluid particles

## References

1. "Real-Time Collision Detection" - Christer Ericson
2. "GPU Gems 3: Chapter 32" - Broad-Phase Collision Detection with CUDA
3. CuPy Documentation: https://docs.cupy.dev/
4. OpenGL Documentation: https://www.opengl.org/

## License

MIT License

## Author

Cindy - Computer Animation Course Project 2025-26

## Acknowledgments

Thanks to:
- CuPy team for excellent GPU acceleration framework
- PyOpenGL community for 3D rendering support
- NVIDIA for CUDA technology

````
