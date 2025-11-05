#!/usr/bin/env python3
"""
简单的验证测试

快速测试系统是否正常工作。

运行方式:
    python tests/simple_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cupy as cp


def test_cupy():
    """测试CuPy是否正常工作"""
    print("Testing CuPy...")
    try:
        # 创建数组
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        
        print(f"  CuPy version: {cp.__version__}")
        print(f"  CUDA available: {cp.cuda.is_available()}")
        print(f"  Device count: {cp.cuda.runtime.getDeviceCount()}")
        print(f"  Test array sum: {c}")
        print("  ✓ CuPy works!")
        return True
    except Exception as e:
        print(f"  ✗ CuPy error: {e}")
        return False


def test_rigid_body():
    """测试刚体系统"""
    print("\nTesting RigidBodySystem...")
    try:
        from src import RigidBodySystem
        
        bodies = RigidBodySystem(100, device_id=0)
        
        # 设置数据
        positions = np.random.rand(100, 3).astype(np.float32)
        bodies.set_positions(positions)
        
        # 检索数据
        data = bodies.to_cpu()
        
        print(f"  Created {len(bodies)} bodies")
        print(f"  Positions shape: {data['positions'].shape}")
        print("  ✓ RigidBodySystem works!")
        return True
    except Exception as e:
        print(f"  ✗ RigidBodySystem error: {e}")
        return False


def test_spatial_grid():
    """测试空间网格"""
    print("\nTesting UniformGrid...")
    try:
        from src import UniformGrid
        
        grid = UniformGrid(
            world_min=(-10, -10, -10),
            world_max=(10, 10, 10),
            cell_size=2.0
        )
        
        # 测试坐标转换
        positions = cp.array([[0, 0, 0], [5, 5, 5]], dtype=cp.float32)
        grid_coords = grid.get_grid_coord(positions)
        hashes = grid.get_grid_hash(grid_coords)
        
        print(f"  Grid resolution: {tuple(cp.asnumpy(grid.resolution))}")
        print(f"  Total cells: {grid.total_cells}")
        print(f"  Test hashes: {cp.asnumpy(hashes)}")
        print("  ✓ UniformGrid works!")
        return True
    except Exception as e:
        print(f"  ✗ UniformGrid error: {e}")
        return False


def test_simulator():
    """测试仿真器"""
    print("\nTesting PhysicsSimulator...")
    try:
        from src import PhysicsSimulator
        
        sim = PhysicsSimulator(
            num_objects=100,
            world_bounds=((-10, -10, -10), (10, 10, 10)),
            cell_size=2.0
        )
        
        # 初始化
        positions = np.random.uniform(-8, 8, (100, 3)).astype(np.float32)
        sim.bodies.set_positions(positions)
        
        velocities = np.random.uniform(-1, 1, (100, 3)).astype(np.float32)
        sim.bodies.set_velocities(velocities)
        
        # 运行几步
        for i in range(5):
            step_info = sim.step()
            print(f"  Step {i+1}: {step_info['num_collisions']} collisions, "
                  f"{step_info['total_time']:.2f}ms")
        
        print("  ✓ PhysicsSimulator works!")
        return True
    except Exception as e:
        print(f"  ✗ PhysicsSimulator error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 70)
    print("GPU Collision Detection - System Verification")
    print("=" * 70)
    
    tests = [
        ("CuPy", test_cupy),
        ("RigidBodySystem", test_rigid_body),
        ("UniformGrid", test_spatial_grid),
        ("PhysicsSimulator", test_simulator),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:25s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
