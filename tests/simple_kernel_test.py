#!/usr/bin/env python3
"""
简单kernel测试
"""

import cupy as cp

# 创建一个简单的测试kernel
TEST_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void test_kernel(
    float* positions,
    float* velocities,
    float gravity_y,
    float dt,
    int num_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // 打印第一个线程的信息
    if (idx == 0) {
        printf("Kernel running! idx=0, gravity_y=%f, dt=%f\\n", gravity_y, dt);
        printf("Before: pos_y=%f, vel_y=%f\\n", positions[1], velocities[1]);
    }
    
    // 更新速度
    float vy = velocities[idx * 3 + 1];
    vy += gravity_y * dt;
    velocities[idx * 3 + 1] = vy;
    
    // 更新位置
    float py = positions[idx * 3 + 1];
    py += vy * dt;
    positions[idx * 3 + 1] = py;
    
    if (idx == 0) {
        printf("After: pos_y=%f, vel_y=%f\\n", positions[1], velocities[1]);
    }
}
''', 'test_kernel')

def main():
    print("Simple Kernel Test")
    print("=" * 50)
    
    N = 5
    positions = cp.zeros((N, 3), dtype=cp.float32)
    positions[:, 1] = 10.0  # All Y=10
    
    velocities = cp.zeros((N, 3), dtype=cp.float32)
    
    gravity_y = -9.8
    dt = 1.0 / 60.0
    
    print(f"\nBefore call:")
    print(f"  pos[0,1]: {positions[0, 1].get()}")
    print(f"  vel[0,1]: {velocities[0, 1].get()}")
    
    # 调用kernel
    TEST_KERNEL(
        (1,), (256,),
        (positions, velocities, gravity_y, dt, N)
    )
    
    cp.cuda.Stream.null.synchronize()
    
    print(f"\nAfter call:")
    print(f"  pos[0,1]: {positions[0, 1].get()}")
    print(f"  vel[0,1]: {velocities[0, 1].get()}")
    
    expected_v = gravity_y * dt
    expected_p = 10.0 + expected_v * dt
    
    print(f"\nExpected:")
    print(f"  vel[0,1]: {expected_v}")
    print(f"  pos[0,1]: {expected_p}")
    
    if abs(velocities[0, 1].get() - expected_v) < 0.01:
        print("\nOK: Kernel works!")
    else:
        print("\nERROR: Kernel problem!")

if __name__ == '__main__':
    main()
