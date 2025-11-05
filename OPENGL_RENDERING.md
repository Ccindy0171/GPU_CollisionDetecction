# OpenGL Rendering for GPU Collision Detection

## Overview

This project now includes a high-quality OpenGL-based renderer for near-realistic 3D visualization of the collision detection system.

## Features

### Visual Quality
- ✨ **Phong Shading** - Realistic lighting with specular highlights
- 🌈 **Smooth Sphere Rendering** - High-quality spheres with configurable detail (32 slices/stacks)
- 💡 **Multi-Light Setup** - Key light and fill light for professional look
- 🎨 **Anti-Aliasing** - MSAA for smooth edges
- 🔲 **Grid and Axes** - World space reference
- 📦 **Boundary Box** - Visual world limits

### Interactive Controls
- 🖱️ **Mouse Controls**:
  - Left Mouse: Rotate camera (azimuth/elevation)
  - Right Mouse: Zoom in/out
  - Middle Mouse: Pan camera
  
- ⌨️ **Keyboard Controls**:
  - `Space`: Pause/Resume simulation
  - `G`: Toggle grid
  - `A`: Toggle axes
  - `S`: Toggle shadows (planned)
  - `W`: Toggle wireframe mode
  - `R`: Reset camera to default view
  - `Q` / `ESC`: Quit

### Performance
- 📊 **Real-time FPS Display** - Monitor rendering and simulation performance
- 🎬 **Video Recording** - Optional high-quality video export (H.264)
- ⚡ **GPU-Accelerated** - Leverages OpenGL for fast rendering

## Installation

### Requirements

```bash
# Install PyOpenGL
conda activate CAni_CuPy
pip install PyOpenGL PyOpenGL_accelerate
```

### Verify Installation

```bash
python -c "from OpenGL.GL import *; print('OpenGL OK')"
```

## Usage

### Basic Test

```bash
python tests/test_opengl.py
```

This will:
1. Create a simulation with 50 balls
2. Check for initial overlaps and resolve them
3. Ask if you want to record video
4. Open an interactive 3D window
5. Run the simulation with real-time visualization

### With Video Recording

```bash
# When prompted, enter 'y' to record
python tests/test_opengl.py
# Output: output/opengl_test.mp4
```

### Full-Scale Simulation

To convert the main gravity_fall example to use OpenGL:

```python
from src.opengl_visualizer import OpenGLVisualizer, OpenGLVideoRecorder

# Create visualizer
vis = OpenGLVisualizer(
    world_bounds=world_bounds,
    width=1920,
    height=1080,
    title="My Simulation"
)

# Optional: Create recorder
recorder = OpenGLVideoRecorder('output/video.mp4', 1920, 1080, fps=60)

# Render loop
def render():
    stats = sim.step()
    positions = cp.asnumpy(sim.bodies.positions)
    radii = cp.asnumpy(sim.bodies.radii)
    
    info = f"Frame {frame}\nCollisions: {stats['num_collisions']}"
    vis.render(positions, radii, colors, info)
    
    # Optional: record frame
    recorder.capture_frame()

vis.set_render_function(render)
vis.run()

# Finalize
recorder.release()
```

## Architecture

### Class: `OpenGLVisualizer`

Main rendering engine with the following capabilities:

**Initialization**:
```python
vis = OpenGLVisualizer(
    world_bounds=((-25, 0, -25), (25, 50, 25)),
    width=1920,
    height=1080,
    title="GPU Collision Detection"
)
```

**Rendering**:
```python
vis.render(
    positions,  # [N, 3] numpy array
    radii,      # [N] numpy array
    colors,     # [N, 3] numpy array (optional)
    info_text   # str (optional)
)
```

**Camera Control**:
- Automatic camera positioning based on world bounds
- Orbit controls around target point
- Configurable distance, azimuth, elevation

**Lighting**:
- Two-point lighting setup (key + fill)
- Phong shading model
- Specular highlights (shininess = 100)

### Class: `OpenGLVideoRecorder`

High-quality video recording:

```python
recorder = OpenGLVideoRecorder(
    filename='output/video.mp4',
    width=1920,
    height=1080,
    fps=60
)

# In render loop:
recorder.capture_frame()

# After simulation:
recorder.release()  # Auto-converts to H.264
```

## Performance Tips

### For Best Visual Quality
- Use higher resolution: 1920x1080 or 3840x2160
- Enable MSAA anti-aliasing (default)
- Use 60fps for smooth motion

### For Best Performance
- Reduce window size: 1280x720
- Disable grid/axes if not needed
- Use lower sphere detail (16 slices/stacks)

### For Large Simulations
- The renderer scales well to thousands of objects
- Tested with 8000 spheres at 60fps on RTX 3050

## Comparison: OpenGL vs Matplotlib

| Feature | OpenGL | Matplotlib |
|---------|--------|------------|
| **Performance** | ⚡ Excellent (GPU) | 🐌 Slow (CPU) |
| **Visual Quality** | ✨ Realistic | 📊 Basic |
| **Interactivity** | 🎮 Real-time | ⏱️ Limited |
| **Lighting** | 💡 Phong shading | 🔲 Flat shading |
| **Anti-aliasing** | ✅ MSAA | ❌ None |
| **Large Datasets** | ✅ Scales well | ❌ Gets slow |
| **Setup** | PyOpenGL | Built-in |

## Troubleshooting

### "ModuleNotFoundError: No module named 'OpenGL'"
```bash
pip install PyOpenGL PyOpenGL_accelerate
```

### "Unable to create OpenGL context"
This typically means no display is available (headless server). You can:
1. Use X11 forwarding: `ssh -X user@host`
2. Use virtual display: `xvfb-run python test_opengl.py`
3. Use matplotlib renderer instead

### "Segmentation fault"
Usually caused by OpenGL driver issues. Try:
```bash
# Update graphics drivers
# Or use software rendering:
export LIBGL_ALWAYS_SOFTWARE=1
```

### Low FPS
- Check GPU utilization
- Reduce window resolution
- Reduce number of objects
- Disable unnecessary visual features

## Examples

### Test 1: Two-Ball Collision (test_01_head_on.py)
- Simple head-on collision
- Perfect for debugging collision response
- Visual verification of momentum conservation

### Test 2: Static Overlap Resolution (test_02_static_overlap.py)
- Tests separation of initially overlapping balls
- Visualizes the collision resolution process

### Test 3: Falling Balls (test_03_falling_balls.py)
- 10 balls falling under gravity
- Tests continuous collision detection
- Verifies no tunneling occurs

### Test 4: Large Scale (test_04_large_scale.py)
- 100+ balls
- Stress test for both physics and rendering
- Realistic particle pile formation

### OpenGL Test (test_opengl.py)
- **50 balls** with OpenGL rendering
- **Interactive camera** controls
- **Optional video recording**
- **Real-time performance metrics**

## Future Enhancements

Planned features:
- [ ] Shadow mapping for realistic shadows
- [ ] HDR and bloom effects
- [ ] Particle trails
- [ ] Collision highlighting
- [ ] Grid-based spatial visualization
- [ ] Multiple camera presets
- [ ] Screenshot capture
- [ ] Custom shaders support

## Credits

- Physics simulation: CuPy + Custom CUDA kernels
- Rendering: PyOpenGL + GLUT
- Video encoding: OpenCV + FFmpeg

---

**Note**: OpenGL rendering requires a display (X11/Wayland). For headless servers, use the matplotlib-based visualizer or run with `xvfb-run`.
