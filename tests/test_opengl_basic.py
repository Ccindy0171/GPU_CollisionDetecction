#!/usr/bin/env python3
"""
Simple test to verify OpenGL visualizer works correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Test 1: Import test
print("Test 1: Importing OpenGL modules...")
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    print("✓ OpenGL modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import OpenGL: {e}")
    sys.exit(1)

# Test 2: Import visualizer
print("\nTest 2: Importing visualizer...")
try:
    from src.opengl_visualizer import OpenGLVisualizer
    print("✓ OpenGLVisualizer imported successfully")
except Exception as e:
    print(f"✗ Failed to import visualizer: {e}")
    sys.exit(1)

# Test 3: Create visualizer instance
print("\nTest 3: Creating visualizer instance...")
try:
    vis = OpenGLVisualizer(
        world_bounds=((-10, 0, -10), (10, 20, 10)),
        width=800,
        height=600,
        title="Test"
    )
    print("✓ Visualizer created successfully")
except Exception as e:
    print(f"✗ Failed to create visualizer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test render with dummy data
print("\nTest 4: Testing render with dummy data...")
try:
    # Create dummy data
    positions = np.array([
        [0, 5, 0],
        [2, 5, 0],
        [-2, 5, 0]
    ], dtype=np.float32)
    
    radii = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Try to render (this will open a window briefly)
    vis.render(positions, radii, colors, "Test render")
    print("✓ Render completed successfully")
    
    # Close immediately
    print("\nClosing window in 2 seconds...")
    import time
    time.sleep(2)
    vis.close()
    
except Exception as e:
    print(f"✗ Render failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed!")
print("="*60)
