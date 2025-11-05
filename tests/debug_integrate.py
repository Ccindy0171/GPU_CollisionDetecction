#!/usr/bin/env python3
"""
Debug INTEGRATE_KERNEL with printf
"""

import cupy as cp
import numpy as np

# Same kernel but with printf debugging
DEBUG_INTEGRATE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void integrate_physics_debug(
    float* positions,
    float* velocities,
    const float* forces,
    const float* masses,
    const float* gravity,
    float dt,
    float damping,
    const float* bounds_min,
    const float* bounds_max,
    const float* radii,
    const float* restitutions,
    int num_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    if (idx == 0) {
        printf("=== Kernel Execution ===\\n");
        printf("idx=%d, num_objects=%d\\n", idx, num_objects);
        printf("dt=%f, damping=%f\\n", dt, damping);
        printf("gravity=[%f, %f, %f]\\n", gravity[0], gravity[1], gravity[2]);
        printf("mass=%f\\n", masses[idx]);
        printf("forces=[%f, %f, %f]\\n", forces[0], forces[1], forces[2]);
        printf("Initial: pos_y=%f, vel_y=%f\\n", positions[1], velocities[1]);
    }
    
    float mass = masses[idx];
    float inv_mass = 1.0f / mass;
    float radius = radii[idx];
    float restitution = restitutions[idx];
    
    // Calculate total acceleration
    float ax = forces[idx * 3 + 0] * inv_mass + gravity[0];
    float ay = forces[idx * 3 + 1] * inv_mass + gravity[1];
    float az = forces[idx * 3 + 2] * inv_mass + gravity[2];
    
    if (idx == 0) {
        printf("Acceleration: ax=%f, ay=%f, az=%f\\n", ax, ay, az);
    }
    
    // Semi-implicit Euler
    float vx = velocities[idx * 3 + 0] + ax * dt;
    float vy = velocities[idx * 3 + 1] + ay * dt;
    float vz = velocities[idx * 3 + 2] + az * dt;
    
    if (idx == 0) {
        printf("After accel: vx=%f, vy=%f, vz=%f\\n", vx, vy, vz);
    }
    
    // Apply damping
    float damping_factor = 1.0f - damping * dt;
    vx *= damping_factor;
    vy *= damping_factor;
    vz *= damping_factor;
    
    // Update position
    float px = positions[idx * 3 + 0] + vx * dt;
    float py = positions[idx * 3 + 1] + vy * dt;
    float pz = positions[idx * 3 + 2] + vz * dt;
    
    if (idx == 0) {
        printf("After integration: px=%f, py=%f, pz=%f\\n", px, py, pz);
    }
    
    // Boundary check (simplified)
    if (py - radius < bounds_min[1]) {
        py = bounds_min[1] + radius;
        vy = -vy * restitution;
    }
    
    // Write back
    positions[idx * 3 + 0] = px;
    positions[idx * 3 + 1] = py;
    positions[idx * 3 + 2] = pz;
    
    velocities[idx * 3 + 0] = vx;
    velocities[idx * 3 + 1] = vy;
    velocities[idx * 3 + 2] = vz;
    
    if (idx == 0) {
        printf("Final: pos_y=%f, vel_y=%f\\n", positions[1], velocities[1]);
        printf("====================\\n");
    }
}
''', 'integrate_physics_debug')

def main():
    print("Debug INTEGRATE_KERNEL")
    print("=" * 60)
    
    N = 1
    positions = cp.array([[0.0, 8.0, 0.0]], dtype=cp.float32)
    velocities = cp.array([[0.0, 0.0, 0.0]], dtype=cp.float32)
    forces = cp.zeros((N, 3), dtype=cp.float32)
    masses = cp.ones(N, dtype=cp.float32)
    radii = cp.ones(N, dtype=cp.float32) * 0.3
    restitutions = cp.ones(N, dtype=cp.float32) * 0.8
    gravity = cp.array([0.0, -9.8, 0.0], dtype=cp.float32)
    bounds_min = cp.array([-10.0, 0.0, -10.0], dtype=cp.float32)
    bounds_max = cp.array([10.0, 20.0, 10.0], dtype=cp.float32)
    
    dt = float(1.0 / 60.0)
    damping = float(0.0)
    
    print("Calling kernel (will print debug info)...\n")
    
    DEBUG_INTEGRATE_KERNEL(
        (1,), (256,),
        (
            positions,
            velocities,
            forces,
            masses,
            gravity,
            dt,
            damping,
            bounds_min,
            bounds_max,
            radii,
            restitutions,
            N
        )
    )
    
    cp.cuda.Stream.null.synchronize()
    
    print(f"\nAfter kernel:")
    print(f"  pos_y: {positions[0, 1].get()}")
    print(f"  vel_y: {velocities[0, 1].get()}")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
