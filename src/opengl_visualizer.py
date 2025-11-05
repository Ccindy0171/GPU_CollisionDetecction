"""
OpenGL-based realistic 3D visualizer for collision detection

Features:
- Phong shading with specular highlights
- Shadows
- Anti-aliasing
- Smooth sphere rendering
- Real-time camera control
"""

import numpy as np
import cupy as cp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
from typing import Tuple, Optional
import time


class Sphere:
    """High-quality sphere using GLU quadric"""
    def __init__(self, slices: int = 32, stacks: int = 32):
        self.slices = slices
        self.stacks = stacks
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
        gluQuadricTexture(self.quadric, GL_TRUE)
    
    def draw(self, radius: float):
        """Draw a sphere at origin with given radius"""
        gluSphere(self.quadric, radius, self.slices, self.stacks)
    
    def __del__(self):
        if hasattr(self, 'quadric') and self.quadric is not None:
            try:
                gluDeleteQuadric(self.quadric)
            except:
                pass  # Ignore cleanup errors during shutdown


class OpenGLVisualizer:
    """
    High-quality OpenGL renderer for physics simulation
    
    Features:
    - Phong lighting model
    - Multiple light sources
    - Smooth shading
    - Anti-aliasing
    - Camera controls (mouse/keyboard)
    """
    
    def __init__(
        self,
        world_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        width: int = 1920,
        height: int = 1080,
        title: str = "GPU Collision Detection"
    ):
        """
        Initialize OpenGL visualizer
        
        Args:
            world_bounds: ((xmin, ymin, zmin), (xmax, ymax, zmax))
            width: Window width
            height: Window height
            title: Window title
        """
        self.world_bounds = world_bounds
        self.width = width
        self.height = height
        self.title = title
        
        # World dimensions
        self.world_min = np.array(world_bounds[0], dtype=np.float32)
        self.world_max = np.array(world_bounds[1], dtype=np.float32)
        self.world_center = (self.world_min + self.world_max) / 2
        self.world_size = np.linalg.norm(self.world_max - self.world_min)
        
        # Camera parameters
        self.camera_distance = self.world_size * 1.5
        self.camera_azimuth = 45.0  # degrees
        self.camera_elevation = 30.0  # degrees
        self.camera_target = self.world_center.copy()
        
        # Mouse interaction
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_button = -1
        
        # Animation state
        self.paused = False
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        # Rendering options
        self.show_grid = True
        self.show_axes = True
        self.show_shadows = True
        self.wireframe = False
        
        # Sphere renderer
        self.sphere = None
        
        # Initialize GLUT and OpenGL
        self._init_glut()
        self._init_gl()
    
    def _init_glut(self):
        """Initialize GLUT window"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(self.title.encode())
        
        # Set up callbacks (will be connected later)
        glutDisplayFunc(self._display_callback)
        glutReshapeFunc(self._reshape_callback)
        glutKeyboardFunc(self._keyboard_callback)
        glutMouseFunc(self._mouse_callback)
        glutMotionFunc(self._motion_callback)
        glutIdleFunc(self._idle_callback)
    
    def _init_gl(self):
        """Initialize OpenGL settings"""
        # Enable features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)  # Anti-aliasing
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Background color (light gray)
        glClearColor(0.95, 0.95, 0.95, 1.0)
        
        # Depth test
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        
        # Set up lighting
        self._setup_lighting()
        
        # Create sphere renderer
        self.sphere = Sphere(slices=32, stacks=32)
        
        # Material properties for spheres
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 100.0)
    
    def _setup_lighting(self):
        """Set up multiple light sources for realistic lighting"""
        # Key light (main directional light from top-right)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Fill light (softer light from left)
        glLightfv(GL_LIGHT1, GL_POSITION, [-0.5, 0.5, 0.5, 0.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
    
    def _display_callback(self):
        """GLUT display callback"""
        if hasattr(self, '_render_func') and self._render_func is not None:
            self._render_func()
    
    def _reshape_callback(self, width: int, height: int):
        """Handle window resize"""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        self._setup_projection()
    
    def _keyboard_callback(self, key: bytes, x: int, y: int):
        """Handle keyboard input"""
        key = key.decode('utf-8').lower()
        
        if key == 'q' or key == '\x1b':  # ESC
            glutLeaveMainLoop()
        elif key == ' ':
            self.paused = not self.paused
        elif key == 'g':
            self.show_grid = not self.show_grid
        elif key == 'a':
            self.show_axes = not self.show_axes
        elif key == 's':
            self.show_shadows = not self.show_shadows
        elif key == 'w':
            self.wireframe = not self.wireframe
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL)
        elif key == 'r':
            # Reset camera
            self.camera_azimuth = 45.0
            self.camera_elevation = 30.0
            self.camera_distance = self.world_size * 1.5
            self.camera_target = self.world_center.copy()
        
        glutPostRedisplay()
    
    def _mouse_callback(self, button: int, state: int, x: int, y: int):
        """Handle mouse button events"""
        if state == GLUT_DOWN:
            self.mouse_button = button
            self.mouse_last_x = x
            self.mouse_last_y = y
        else:
            self.mouse_button = -1
    
    def _motion_callback(self, x: int, y: int):
        """Handle mouse motion"""
        if self.mouse_button == GLUT_LEFT_BUTTON:
            # Rotate camera
            dx = x - self.mouse_last_x
            dy = y - self.mouse_last_y
            
            self.camera_azimuth += dx * 0.5
            self.camera_elevation = np.clip(self.camera_elevation - dy * 0.5, -89, 89)
            
        elif self.mouse_button == GLUT_RIGHT_BUTTON:
            # Zoom
            dy = y - self.mouse_last_y
            self.camera_distance *= (1.0 + dy * 0.01)
            self.camera_distance = np.clip(
                self.camera_distance,
                self.world_size * 0.5,
                self.world_size * 5.0
            )
        
        elif self.mouse_button == GLUT_MIDDLE_BUTTON:
            # Pan
            dx = (x - self.mouse_last_x) * self.camera_distance * 0.001
            dy = (y - self.mouse_last_y) * self.camera_distance * 0.001
            
            # Convert screen space to world space
            azimuth_rad = np.radians(self.camera_azimuth)
            right = np.array([np.cos(azimuth_rad), 0, -np.sin(azimuth_rad)])
            up = np.array([0, 1, 0])
            
            self.camera_target += right * dx - up * dy
        
        self.mouse_last_x = x
        self.mouse_last_y = y
        glutPostRedisplay()
    
    def _idle_callback(self):
        """GLUT idle callback"""
        if not self.paused:
            glutPostRedisplay()
    
    def _setup_projection(self):
        """Set up projection matrix"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = self.width / self.height if self.height > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, self.world_size * 10.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def _setup_camera(self):
        """Set up camera view matrix"""
        glLoadIdentity()
        
        # Calculate camera position
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        
        cam_x = self.camera_target[0] + self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        cam_y = self.camera_target[1] + self.camera_distance * np.sin(elevation_rad)
        cam_z = self.camera_target[2] + self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        
        gluLookAt(
            cam_x, cam_y, cam_z,  # Camera position
            self.camera_target[0], self.camera_target[1], self.camera_target[2],  # Look at
            0.0, 1.0, 0.0  # Up vector
        )
    
    def _draw_grid(self):
        """Draw ground grid"""
        if not self.show_grid:
            return
        
        glDisable(GL_LIGHTING)
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.0)
        
        # Draw grid on ground plane (y = ymin)
        y = self.world_min[1]
        x_min, x_max = self.world_min[0], self.world_max[0]
        z_min, z_max = self.world_min[2], self.world_max[2]
        
        grid_size = 5.0
        
        glBegin(GL_LINES)
        
        # X lines
        x = x_min
        while x <= x_max:
            glVertex3f(x, y, z_min)
            glVertex3f(x, y, z_max)
            x += grid_size
        
        # Z lines
        z = z_min
        while z <= z_max:
            glVertex3f(x_min, y, z)
            glVertex3f(x_max, y, z)
            z += grid_size
        
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _draw_axes(self):
        """Draw coordinate axes"""
        if not self.show_axes:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        
        axis_length = self.world_size * 0.15
        origin = self.world_min
        
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0] + axis_length, origin[1], origin[2])
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0], origin[1] + axis_length, origin[2])
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0], origin[1], origin[2] + axis_length)
        
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_boundary_box(self):
        """Draw world boundary box"""
        glDisable(GL_LIGHTING)
        glColor4f(0.5, 0.5, 0.5, 0.3)
        glLineWidth(2.0)
        
        x_min, y_min, z_min = self.world_min
        x_max, y_max, z_max = self.world_max
        
        glBegin(GL_LINES)
        
        # Bottom face
        glVertex3f(x_min, y_min, z_min); glVertex3f(x_max, y_min, z_min)
        glVertex3f(x_max, y_min, z_min); glVertex3f(x_max, y_min, z_max)
        glVertex3f(x_max, y_min, z_max); glVertex3f(x_min, y_min, z_max)
        glVertex3f(x_min, y_min, z_max); glVertex3f(x_min, y_min, z_min)
        
        # Top face
        glVertex3f(x_min, y_max, z_min); glVertex3f(x_max, y_max, z_min)
        glVertex3f(x_max, y_max, z_min); glVertex3f(x_max, y_max, z_max)
        glVertex3f(x_max, y_max, z_max); glVertex3f(x_min, y_max, z_max)
        glVertex3f(x_min, y_max, z_max); glVertex3f(x_min, y_max, z_min)
        
        # Vertical edges
        glVertex3f(x_min, y_min, z_min); glVertex3f(x_min, y_max, z_min)
        glVertex3f(x_max, y_min, z_min); glVertex3f(x_max, y_max, z_min)
        glVertex3f(x_max, y_min, z_max); glVertex3f(x_max, y_max, z_max)
        glVertex3f(x_min, y_min, z_max); glVertex3f(x_min, y_max, z_max)
        
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def render(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        colors: Optional[np.ndarray] = None,
        info_text: Optional[str] = None
    ):
        """
        Render the scene
        
        Args:
            positions: Object positions [N, 3]
            radii: Object radii [N]
            colors: Object colors [N, 3], optional
            info_text: Information text to display
        """
        # Convert CuPy arrays to NumPy if needed
        if hasattr(positions, 'get'):
            positions = positions.get()
        if hasattr(radii, 'get'):
            radii = radii.get()
        if colors is not None and hasattr(colors, 'get'):
            colors = colors.get()
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up projection and camera
        self._setup_projection()
        self._setup_camera()
        
        # Draw scene elements
        self._draw_grid()
        self._draw_axes()
        self._draw_boundary_box()
        
        # Draw spheres
        self._draw_spheres(positions, radii, colors)
        
        # Draw HUD
        if info_text:
            self._draw_hud(info_text)
        
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = current_time
        self.frame_count += 1
        
        # Swap buffers
        glutSwapBuffers()
    
    def _draw_spheres(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        colors: Optional[np.ndarray] = None
    ):
        """Draw all spheres with proper materials"""
        n = len(positions)
        
        for i in range(n):
            pos = positions[i]
            radius = radii[i]
            
            # Set color
            if colors is not None:
                color = colors[i]
            else:
                # Default color based on index
                hue = (i * 0.618033988749895) % 1.0  # Golden ratio
                color = self._hsv_to_rgb(hue, 0.8, 0.9)
            
            # Set material color
            glColor3f(color[0], color[1], color[2])
            
            # Draw sphere at position
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            self.sphere.draw(radius)
            glPopMatrix()
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """Convert HSV to RGB"""
        import colorsys
        return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)
    
    def _draw_hud(self, info_text: str):
        """Draw HUD overlay with information"""
        # Switch to orthographic projection for HUD
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for text
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Draw semi-transparent background
        glColor4f(0.0, 0.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(10, self.height - 10)
        glVertex2f(400, self.height - 10)
        glVertex2f(400, self.height - 150)
        glVertex2f(10, self.height - 150)
        glEnd()
        
        # Draw text
        glColor3f(1.0, 1.0, 1.0)
        y_offset = self.height - 30
        
        # FPS
        self._draw_text(f"FPS: {self.fps:.1f}", 20, y_offset)
        y_offset -= 20
        
        # Info text
        for line in info_text.split('\n'):
            self._draw_text(line, 20, y_offset)
            y_offset -= 20
        
        # Controls hint
        y_offset = 80
        glColor4f(0.0, 0.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(250, 10)
        glVertex2f(250, y_offset)
        glVertex2f(10, y_offset)
        glEnd()
        
        glColor3f(0.8, 0.8, 0.8)
        y_offset = 70
        self._draw_text("Controls:", 20, y_offset, scale=0.12)
        y_offset -= 15
        self._draw_text("Left Mouse: Rotate", 20, y_offset, scale=0.10)
        y_offset -= 15
        self._draw_text("Right Mouse: Zoom", 20, y_offset, scale=0.10)
        y_offset -= 15
        self._draw_text("Space: Pause", 20, y_offset, scale=0.10)
        y_offset -= 15
        self._draw_text("G/A/S/W: Toggle features", 20, y_offset, scale=0.10)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def _draw_text(self, text: str, x: float, y: float, scale: float = 0.15):
        """Draw text using GLUT bitmap font"""
        glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))
    
    def set_render_function(self, func):
        """Set the function to be called on each render"""
        self._render_func = func
    
    def run(self):
        """Start the main loop"""
        glutMainLoop()
    
    def close(self):
        """Close the window"""
        glutLeaveMainLoop()


# For video recording with OpenGL
class OpenGLVideoRecorder:
    """Record OpenGL frames to video"""
    
    def __init__(self, filename: str, width: int, height: int, fps: int = 60):
        import cv2
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        temp_filename = filename.replace('.mp4', '_temp.avi')
        self.writer = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
        self.temp_filename = temp_filename
        self.frame_count = 0
        
        print(f"Video recorder initialized: {filename} ({width}x{height} @ {fps}fps)")
    
    def capture_frame(self):
        """Capture current OpenGL frame"""
        import cv2
        
        # Read pixels from OpenGL
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        
        # Flip vertically (OpenGL origin is bottom-left)
        image = np.flipud(image)
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Write frame
        self.writer.write(image)
        self.frame_count += 1
    
    def release(self):
        """Finalize video"""
        import cv2
        import subprocess
        import os
        
        self.writer.release()
        print(f"Video recording completed: {self.frame_count} frames")
        
        # Convert to H.264
        try:
            print("Converting to H.264...")
            cmd = [
                'ffmpeg', '-y',
                '-i', self.temp_filename,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-loglevel', 'error',
                self.filename
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(self.temp_filename)
            print(f"Video converted: {self.filename}")
        except Exception as e:
            print(f"Warning: Could not convert to H.264: {e}")
            print(f"Temporary file saved as: {self.temp_filename}")
