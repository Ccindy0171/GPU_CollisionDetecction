# Demo Instructions and Examples

This document provides detailed instructions for running all examples and demos in the GPU Collision Detection system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Example Programs](#example-programs)
3. [Test Suite](#test-suite)
4. [Configuration Options](#configuration-options)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites Check

Before running any demos, verify your environment:

```bash
# Check Python version (3.8+ required)
python --version

# Check CUDA installation
nvcc --version

# Verify CuPy installation
python -c "import cupy as cp; print('CuPy:', cp.__version__); print('CUDA available:', cp.cuda.is_available())"

# Verify OpenGL installation (for visualization)
python -c "import OpenGL; print('OpenGL version:', OpenGL.__version__)"
```

### Basic Demo (No Visualization)

Run a simple physics-only test:

```bash
cd GPU_CollisionDetecction
python tests/test_physics_only.py
```

Expected output:
```
=================================================================
PHYSICS-ONLY TEST
=================================================================
Testing simulator without OpenGL visualization
Simulating 100 objects for 120 frames...

Frame 0: 0 collisions, 0.00 KE
Frame 10: 15 collisions, 245.67 KE
Frame 20: 42 collisions, 892.34 KE
...
Test completed successfully!
Average FPS: 156.2
```

---

## Example Programs

### 1. Gravity Fall Demo (Main Example)

**Description**: Simulates spheres falling under gravity, colliding, and forming a pile.

**Features**:
- Real-time 3D visualization with OpenGL
- Interactive camera controls
- High-quality video recording
- Performance statistics overlay

**Run Command**:
```bash
python examples/gravity_fall.py
```

**Interactive Controls**:
| Key/Mouse | Action |
|-----------|--------|
| **Left Mouse + Drag** | Rotate camera |
| **Right Mouse + Drag** | Zoom in/out |
| **Middle Mouse + Drag** | Pan camera |
| **Space** | Pause/Resume simulation |
| **G** | Toggle grid display |
| **A** | Toggle axis display |
| **W** | Toggle wireframe mode |
| **R** | Reset camera to default |
| **Q** or **ESC** | Quit program |

**Configuration** (edit in script):
```python
# In examples/gravity_fall.py

# Number of spheres
NUM_OBJECTS = 500           # Range: 10-10000

# World boundaries (meters)
WORLD_BOUNDS = ((-20, 0, -20), (20, 40, 20))

# Grid cell size (should be ~2x average radius)
CELL_SIZE = 2.0

# Simulation duration
NUM_FRAMES = 600            # 10 seconds @ 60fps

# Video recording
RECORD_VIDEO = True         # Set to False to disable
VIDEO_PATH = "output/gravity_fall.mp4"
```

**Expected Output**:
- Real-time visualization window (1920×1080)
- Console output showing FPS and collision count
- Video file: `output/gravity_fall.mp4` (if recording enabled)

**Performance**:
- RTX 3050: 150-250 FPS (500 objects)
- RTX 3060: 200-350 FPS (500 objects)
- GTX 1660: 80-150 FPS (500 objects)

---

### 2. Quick Test Demo

**Description**: Minimal test for quick verification.

**Run Command**:
```bash
python examples/quick_test.py
```

**What it does**:
- Creates 50 spheres in a small space
- Runs 60 frames (1 second)
- Prints performance metrics
- No visualization (headless)

**Expected Output**:
```
Quick Test - 50 objects
Frame time: 2.1ms
FPS: 476
Total collisions: 127
✓ Test passed
```

**Use Case**: Quick smoke test, CI/CD pipeline verification

---

## Test Suite

### Unit Tests

#### Test 1: Head-On Collision

**Description**: Two spheres collide head-on, validates impulse physics.

**Run Command**:
```bash
python tests/test_01_head_on.py
```

**Expected Behavior**:
1. Two spheres approach each other at equal speeds
2. They collide and bounce back
3. Velocities reverse (elastic collision)
4. Kinetic energy conserved (within 5%)

**Pass Criteria**:
- Objects separate after collision
- No penetration remaining
- Momentum conserved

---

#### Test 2: Static Overlap Resolution

**Description**: Tests resolution of initially overlapping spheres.

**Run Command**:
```bash
python tests/test_02_static_overlap.py
```

**Expected Behavior**:
1. Two spheres placed overlapping
2. Position correction separates them
3. No residual penetration after 5 frames

**Pass Criteria**:
- Penetration depth < 0.01m within 5 frames
- Objects pushed apart proportional to mass

---

#### Test 3: Multiple Falling Balls

**Description**: Tests multi-body interactions with gravity.

**Run Command**:
```bash
python tests/test_03_falling_balls.py
```

**Configuration**:
- 20 balls arranged in layers
- Gravity: -9.81 m/s²
- Duration: 300 frames (5 seconds)

**Expected Behavior**:
- Balls fall and collide
- Form stable pile at bottom
- No balls escape boundaries
- No excessive penetration

---

#### Test 4: Large Scale Test

**Description**: Stress test with 100+ objects.

**Run Command**:
```bash
python tests/test_04_large_scale.py
```

**Configurable via Arguments**:
```bash
# Custom object count
python tests/test_04_large_scale.py --objects 500

# Custom duration
python tests/test_04_large_scale.py --frames 1000

# Custom world size
python tests/test_04_large_scale.py --world-size 100
```

**Performance Targets**:
| Objects | Target FPS | Expected Collisions |
|---------|------------|---------------------|
| 100     | >120       | ~50-100             |
| 500     | >30        | ~300-600            |
| 1000    | >20        | ~500-1000           |

---

### Functional Tests

#### OpenGL Rendering Test

**Description**: Validates visualization without physics.

**Run Command**:
```bash
python tests/test_opengl_basic.py
```

**What it tests**:
- OpenGL context creation
- Sphere rendering
- Lighting and shading
- Camera controls

**Expected Output**:
- Visualization window appears
- Spheres rendered with Phong shading
- Smooth rotation with mouse

---

#### Benchmark Suite

**Description**: Comprehensive performance benchmarking.

**Run Command**:
```bash
python tests/benchmark.py
```

**What it benchmarks**:
1. Scalability: 500, 1000, 2000, 5000, 10000 objects
2. Cell size optimization: 0.5m to 5.0m
3. Component timings: grid, detection, response, integration
4. GPU utilization metrics

**Output**:
- CSV file: `output/benchmark_results.csv`
- Performance plots: `output/benchmark_plots.png`
- Summary report printed to console

**Duration**: ~15-20 minutes

---

## Configuration Options

### Simulator Parameters

```python
sim = PhysicsSimulator(
    num_objects=1000,           # Number of rigid bodies
    world_bounds=(              # Simulation space boundaries
        (-50, 0, -50),          # (xmin, ymin, zmin) in meters
        (50, 50, 50)            # (xmax, ymax, zmax)
    ),
    cell_size=2.0,              # Spatial grid cell size (meters)
    device_id=0,                # GPU device index (0 for primary)
    dt=1.0/60.0,                # Time step (seconds)
    gravity=(0, -9.81, 0),      # Gravity vector (m/s²)
    damping=0.01                # Velocity damping coefficient [0,1]
)
```

### Object Properties

```python
# Set radii (sphere sizes)
sim.bodies.set_radii(np.random.uniform(0.3, 0.7, num_objects))

# Set masses (kg)
sim.bodies.set_masses(np.random.uniform(1.0, 5.0, num_objects))

# Set restitution (bounciness: 0=inelastic, 1=perfectly elastic)
sim.bodies.set_restitutions(np.random.uniform(0.5, 0.9, num_objects))

# Set colors (RGB, range [0,1])
colors = np.random.rand(num_objects, 3)
sim.bodies.set_colors(colors)
```

### Visualization Options

```python
vis = OpenGLVisualizer(
    world_bounds=world_bounds,
    width=1920,                 # Window width (pixels)
    height=1080,                # Window height (pixels)
    title="My Simulation"
)

# Recording settings
recorder = OpenGLVideoRecorder(
    visualizer=vis,
    output_path="output/my_video.mp4",
    fps=60,                     # Video framerate
    codec='h264',               # Video codec
    bitrate='5M'                # Video bitrate
)
```

### Performance Tuning

```python
# For better performance on low-end GPUs
sim = PhysicsSimulator(
    num_objects=500,            # Reduce object count
    cell_size=3.0,              # Increase cell size (fewer cells)
    dt=1.0/30.0,                # Lower framerate target
    damping=0.05                # Higher damping (faster stabilization)
)

# For maximum quality on high-end GPUs
sim = PhysicsSimulator(
    num_objects=10000,          # More objects
    cell_size=1.5,              # Finer grid
    dt=1.0/120.0,               # Higher precision
    damping=0.001               # Lower damping (more realistic)
)
```

---

## Troubleshooting

### Common Issues

#### 1. "CUDA_ERROR_OUT_OF_MEMORY"

**Cause**: Not enough GPU memory for the number of objects.

**Solutions**:
```python
# Reduce object count
NUM_OBJECTS = 500  # Instead of 5000

# Increase cell size (reduces grid memory)
CELL_SIZE = 3.0    # Instead of 2.0

# Disable visualization (saves VRAM)
RECORD_VIDEO = False
```

**VRAM Requirements**:
- 500 objects: ~200 MB
- 1000 objects: ~350 MB
- 5000 objects: ~1.5 GB
- 10000 objects: ~2.8 GB

---

#### 2. "ImportError: No module named 'cupy'"

**Cause**: CuPy not installed or wrong CUDA version.

**Solution**:
```bash
# Check CUDA version
nvcc --version

# Install matching CuPy
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install cupy-cuda11x  # For CUDA 11.x
```

---

#### 3. Low Frame Rate / Stuttering

**Causes & Solutions**:

**Cause: Too many objects**
```python
# Reduce object count
NUM_OBJECTS = 1000  # Target: 30+ FPS
```

**Cause: Cell size too small**
```python
# Increase cell size
CELL_SIZE = 2.5  # Reduce grid resolution
```

**Cause: High collision rate**
```python
# Increase world size (reduce density)
WORLD_BOUNDS = ((-100, 0, -100), (100, 50, 100))
```

**Cause: Visualization overhead**
```python
# Disable OpenGL or use lower resolution
vis = OpenGLVisualizer(world_bounds, width=1280, height=720)
```

---

#### 4. "OpenGL: Display not found" (Linux)

**Cause**: X11 display not available (headless server).

**Solution**:
```bash
# Use Xvfb (virtual framebuffer)
sudo apt-get install xvfb
xvfb-run python examples/gravity_fall.py

# OR disable visualization
python tests/test_physics_only.py
```

---

#### 5. Objects Falling Through Floor / Tunneling

**Cause**: Time step too large for object velocities.

**Solution**:
```python
# Reduce time step
dt=1.0/120.0  # Instead of 1/60

# OR reduce gravity
gravity=(0, -5.0, 0)  # Instead of -9.81

# OR increase boundary restitution
# (increases bounce, slows down impact)
```

---

#### 6. Poor Collision Detection (Objects Pass Through)

**Cause**: Cell size too large or objects too fast.

**Solutions**:
```python
# Reduce cell size (more accurate)
CELL_SIZE = 1.5  # Instead of 2.0

# Reduce time step (smaller integration steps)
dt=1.0/120.0

# Add velocity clamping
max_velocity = 50.0  # m/s
velocities = np.clip(velocities, -max_velocity, max_velocity)
```

---

### Performance Debugging

#### Enable Profiling

```python
from src.performance import PerformanceMonitor

monitor = PerformanceMonitor()

for frame in range(100):
    monitor.start_event('frame')
    stats = sim.step()
    monitor.end_event('frame')

# Print statistics
monitor.print_statistics()

# Generate plots
monitor.plot_timeline(save_path='performance.png')
```

#### Check GPU Utilization

```bash
# Monitor GPU usage in real-time
nvidia-smi -l 1

# Look for:
# - GPU Utilization: Should be 70-100% during simulation
# - Memory Usage: Should be stable (no leaks)
# - Temperature: Should be < 80°C
```

---

## Advanced Usage

### Custom Physics Scenarios

#### Explosion Effect

```python
# Set initial velocities radiating from center
center = np.array([0, 25, 0])
positions = sim.bodies.positions.get()
directions = positions - center
distances = np.linalg.norm(directions, axis=1, keepdims=True)
velocities = (directions / distances) * 20.0  # 20 m/s outward
sim.bodies.set_velocities(velocities)
```

#### Rotating System

```python
# Apply tangential velocities for rotation
positions = sim.bodies.positions.get()
angular_velocity = 2.0  # rad/s
velocities = np.cross([0, angular_velocity, 0], positions)
sim.bodies.set_velocities(velocities)
```

#### Layered Drop

```python
# Arrange objects in horizontal layers, drop sequentially
for layer in range(num_layers):
    # Set positions for this layer
    layer_positions = generate_layer(layer_height)
    sim.bodies.positions[layer*objects_per_layer:(layer+1)*objects_per_layer] = layer_positions
```

---

### Batch Processing

Run multiple simulations without user interaction:

```python
import os

configurations = [
    {'objects': 500, 'cell_size': 2.0},
    {'objects': 1000, 'cell_size': 2.0},
    {'objects': 2000, 'cell_size': 2.5},
]

for config in configurations:
    sim = PhysicsSimulator(num_objects=config['objects'], cell_size=config['cell_size'])
    
    # Run simulation
    for frame in range(600):
        stats = sim.step()
    
    # Save results
    results_path = f"output/sim_{config['objects']}_objects.csv"
    # ... save data ...
```

---

## Expected Outputs

### Console Output

Typical console output during simulation:

```
==============================================================
GPU Collision Detection - Gravity Fall (OpenGL)
==============================================================

Configuration:
  Objects: 500
  World Bounds: ((-20, 0, -20), (20, 40, 20))
  Cell Size: 2.0
  Total Frames: 600
  Duration: 10.0 seconds @ 60fps

Initializing simulator...
  ✓ Simulator created

Setting up scene...
  Radius range: 0.31 - 0.68 m
  Generating 500 non-overlapping spheres...
  ✓ Generated positions successfully
  ✓ Verified no overlaps

Running simulation...
Frame 0: 0 collisions, 4.2ms (238 FPS)
Frame 60: 287 collisions, 4.5ms (222 FPS)
Frame 120: 534 collisions, 4.8ms (208 FPS)
...

Simulation complete!
Average FPS: 215.3
Total collisions: 95,847
Video saved to: output/gravity_fall.mp4
```

### Video Output

- **Format**: MP4 (H.264)
- **Resolution**: 1920×1080 (default)
- **Framerate**: 60 FPS
- **Duration**: 10 seconds (default)
- **Size**: ~20-50 MB (depends on complexity)

### Performance Plots

Benchmark scripts generate visualizations:

- **Timeline plots**: Frame time over simulation
- **Distribution plots**: Histogram of frame times
- **Component breakdown**: Pie chart of time spent per component
- **Scalability plots**: Frame time vs. object count

---

## Additional Resources

### Example Gallery

- **Gravity Fall**: `examples/gravity_fall.py`
- **OpenGL Visualization**: `examples/gravity_fall_opengl.py`
- **Quick Test**: `examples/quick_test.py`

### Test Suite

- **Unit Tests**: `tests/test_01_head_on.py` through `tests/test_04_large_scale.py`
- **Integration Tests**: `tests/test_opengl_basic.py`, `tests/test_physics_only.py`
- **Benchmarks**: `tests/benchmark.py`

### Documentation

- **Architecture**: `docs/ARCHITECTURE.md`
- **Experiment Report**: `docs/EXPERIMENT_REPORT.md`
- **API Reference**: See docstrings in source files

---

## Support

**Issues**: https://github.com/Ccindy0171/GPU_CollisionDetecction/issues  
**Documentation**: `docs/` directory  
**Examples**: `examples/` directory
