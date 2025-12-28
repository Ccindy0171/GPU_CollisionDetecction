#!/usr/bin/env python3
"""
System Check Utility

Validates the environment and provides diagnostics for running
GPU Collision Detection simulations.

Usage:
    python check_system.py
    
This script checks:
- Python version
- CUDA availability
- CuPy installation
- GPU information
- OpenGL support
- Required dependencies
"""

import sys
import platform
import subprocess


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python_version():
    """Check Python version."""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    else:
        print("✓ Python version is compatible")
        return True


def check_cuda():
    """Check CUDA installation."""
    print_section("CUDA Installation")
    
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse CUDA version from output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(line.strip())
            print("✓ CUDA compiler (nvcc) found")
            return True
        else:
            print("❌ CUDA compiler not found")
            return False
    except FileNotFoundError:
        print("❌ CUDA not installed or not in PATH")
        print("   Install CUDA from: https://developer.nvidia.com/cuda-downloads")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_nvidia_driver():
    """Check NVIDIA driver."""
    print_section("NVIDIA Driver")
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse driver version and GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(line.strip())
                elif line.strip() and '|' in line and 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                    # GPU info line
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 2:
                        print(f"GPU: {parts[1]}")
            print("✓ NVIDIA driver installed")
            return True
        else:
            print("❌ nvidia-smi failed to run")
            return False
    except FileNotFoundError:
        print("❌ NVIDIA driver not installed or nvidia-smi not in PATH")
        print("   Install from: https://www.nvidia.com/Download/index.aspx")
        return False
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")
        return False


def check_cupy():
    """Check CuPy installation and GPU availability."""
    print_section("CuPy (GPU Computing Library)")
    
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        
        if cp.cuda.is_available():
            print("✓ CuPy successfully loaded")
            print("✓ CUDA is available")
            
            # Get device info
            device = cp.cuda.Device()
            print(f"\nGPU Device Information:")
            print(f"  Device ID: {device.id}")
            print(f"  Name: {device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name}")
            print(f"  Compute Capability: {device.compute_capability}")
            
            # Memory info
            mem_info = device.mem_info
            free_mem_gb = mem_info[0] / (1024**3)
            total_mem_gb = mem_info[1] / (1024**3)
            print(f"  Memory: {free_mem_gb:.2f} GB free / {total_mem_gb:.2f} GB total")
            
            # CUDA runtime version
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            print(f"  CUDA Runtime: {major}.{minor}")
            
            return True
        else:
            print("❌ CUDA not available to CuPy")
            print("   Check CUDA installation")
            return False
            
    except ImportError:
        print("❌ CuPy not installed")
        print("   Install with: pip install cupy-cuda12x  (for CUDA 12.x)")
        print("              or: pip install cupy-cuda11x  (for CUDA 11.x)")
        return False
    except Exception as e:
        print(f"❌ Error checking CuPy: {e}")
        return False


def check_numpy():
    """Check NumPy installation."""
    print_section("NumPy")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        print("✓ NumPy installed")
        return True
    except ImportError:
        print("❌ NumPy not installed")
        print("   Install with: pip install numpy")
        return False


def check_opengl():
    """Check OpenGL availability."""
    print_section("OpenGL (Visualization)")
    
    try:
        import OpenGL
        print(f"PyOpenGL version: {OpenGL.__version__}")
        
        # Try to import GL, GLU, GLUT
        try:
            from OpenGL.GL import *
            print("✓ OpenGL.GL available")
        except Exception as e:
            print(f"❌ OpenGL.GL import failed: {e}")
            return False
        
        try:
            from OpenGL.GLU import *
            print("✓ OpenGL.GLU available")
        except Exception as e:
            print(f"❌ OpenGL.GLU import failed: {e}")
            return False
        
        try:
            from OpenGL.GLUT import *
            print("✓ OpenGL.GLUT available")
        except Exception as e:
            print(f"⚠  OpenGL.GLUT import failed: {e}")
            print("   GLUT is optional for some visualization features")
        
        print("✓ PyOpenGL installed and functional")
        return True
        
    except ImportError:
        print("❌ PyOpenGL not installed")
        print("   Install with: pip install PyOpenGL PyOpenGL_accelerate")
        print("   Note: Visualization features will not work without OpenGL")
        return False


def check_opencv():
    """Check OpenCV for video recording."""
    print_section("OpenCV (Video Recording)")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print("✓ OpenCV installed")
        return True
    except ImportError:
        print("❌ OpenCV not installed")
        print("   Install with: pip install opencv-python")
        print("   Note: Video recording will not work without OpenCV")
        return False


def check_platform():
    """Display platform information."""
    print_section("Platform Information")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Implementation: {platform.python_implementation()}")


def run_quick_test():
    """Run a quick GPU computation test."""
    print_section("Quick GPU Test")
    
    try:
        import cupy as cp
        import numpy as np
        
        print("Running simple GPU computation...")
        
        # Create arrays on GPU
        a_gpu = cp.random.rand(1000, 1000, dtype=cp.float32)
        b_gpu = cp.random.rand(1000, 1000, dtype=cp.float32)
        
        # Matrix multiplication on GPU
        import time
        start = time.time()
        c_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        
        # Same computation on CPU
        a_cpu = cp.asnumpy(a_gpu)
        b_cpu = cp.asnumpy(b_gpu)
        
        start = time.time()
        c_cpu = np.dot(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        print(f"GPU time: {gpu_time*1000:.2f} ms")
        print(f"CPU time: {cpu_time*1000:.2f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        print("✓ GPU computation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("  GPU Collision Detection - System Check")
    print("=" * 70)
    
    # Run all checks
    checks = {
        'Python Version': check_python_version(),
        'Platform Info': check_platform() or True,  # Always passes
        'CUDA': check_cuda(),
        'NVIDIA Driver': check_nvidia_driver(),
        'CuPy': check_cupy(),
        'NumPy': check_numpy(),
        'OpenGL': check_opengl(),
        'OpenCV': check_opencv(),
    }
    
    # Run quick test if basics pass
    if checks['CuPy'] and checks['NumPy']:
        checks['GPU Test'] = run_quick_test()
    
    # Summary
    print_section("Summary")
    
    essential_checks = ['Python Version', 'CUDA', 'CuPy', 'NumPy']
    optional_checks = ['OpenGL', 'OpenCV']
    
    essential_passed = all(checks.get(c, False) for c in essential_checks)
    optional_passed = [c for c in optional_checks if checks.get(c, False)]
    optional_failed = [c for c in optional_checks if not checks.get(c, False)]
    
    if essential_passed:
        print("✓ All essential components installed correctly")
        print("  You can run physics simulations (headless mode)")
        
        if optional_passed:
            print(f"✓ Optional components available: {', '.join(optional_passed)}")
        
        if optional_failed:
            print(f"⚠  Optional components missing: {', '.join(optional_failed)}")
            print("  Some features (visualization, recording) may not work")
        
        if len(optional_passed) == len(optional_checks):
            print("\n✓ READY: All components installed - full functionality available")
        else:
            print("\n✓ READY: Core functionality available (limited features)")
        
        return 0
    else:
        failed = [c for c in essential_checks if not checks.get(c, False)]
        print(f"❌ Missing essential components: {', '.join(failed)}")
        print("\nPlease install missing components before running simulations.")
        print("See README.md for installation instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
