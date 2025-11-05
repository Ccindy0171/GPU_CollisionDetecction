"""
CUDA内核函数模块

该模块包含所有GPU并行计算的CUDA kernel函数。
使用CuPy的RawKernel接口直接编写CUDA C代码。

所有kernel函数针对RTX 3050优化（4GB显存，2048个CUDA核心）。
"""

import cupy as cp


# ============================================================================
# 网格构建相关Kernel
# ============================================================================

COMPUTE_GRID_HASH_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_grid_hash(
    const float* positions,     // [N, 3] object positions
    int* grid_hashes,           // [N] output: grid hash values
    const float* world_min,     // [3] world space minimum bounds
    float cell_size,            // grid cell size
    const int* resolution,      // [3] grid resolution
    int num_objects             // number of objects
) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // Read object position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    
    // Calculate grid coordinates
    int gx = (int)((px - world_min[0]) / cell_size);
    int gy = (int)((py - world_min[1]) / cell_size);
    int gz = (int)((pz - world_min[2]) / cell_size);
    
    // Clamp to bounds
    gx = max(0, min(gx, resolution[0] - 1));
    gy = max(0, min(gy, resolution[1] - 1));
    gz = max(0, min(gz, resolution[2] - 1));
    
    // Calculate 1D hash value (row-major order)
    grid_hashes[idx] = gz * resolution[1] * resolution[0] + 
                       gy * resolution[0] + gx;
}
''', 'compute_grid_hash')


FIND_CELL_START_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void find_cell_start(
    const int* sorted_hashes,   // [N] sorted grid hash values
    int* cell_starts,           // [total_cells] output: cell start indices
    int* cell_ends,             // [total_cells] output: cell end indices
    int num_objects             // number of objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    int hash = sorted_hashes[idx];
    
    // Check if this is the first element in the cell
    if (idx == 0 || sorted_hashes[idx - 1] != hash) {
        cell_starts[hash] = idx;
    }
    
    // Check if this is the last element in the cell
    if (idx == num_objects - 1 || sorted_hashes[idx + 1] != hash) {
        cell_ends[hash] = idx + 1;  // end index is exclusive
    }
}
''', 'find_cell_start')


# ============================================================================
# 碰撞检测相关Kernel
# ============================================================================

BROAD_PHASE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void broad_phase_collision(
    const float* positions,         // [N, 3] object positions
    const float* radii,             // [N] object radii
    const int* cell_starts,         // [total_cells] cell start indices
    const int* cell_ends,           // [total_cells] cell end indices
    const int* resolution,          // [3] grid resolution
    float cell_size,                // grid cell size
    const float* world_min,         // [3] world space minimum bounds
    int* collision_pairs,           // [max_pairs, 2] output: collision pairs
    int* pair_count,                // [1] output: collision pair count (atomic)
    int num_objects,                // number of objects
    int max_pairs                   // maximum collision pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // Read current object info
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    float r1 = radii[idx];
    
    // Calculate grid coordinates
    int gx = (int)((px - world_min[0]) / cell_size);
    int gy = (int)((py - world_min[1]) / cell_size);
    int gz = (int)((pz - world_min[2]) / cell_size);
    
    // Iterate through 27 neighboring cells (3x3x3 neighborhood)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = gx + dx;
                int ny = gy + dy;
                int nz = gz + dz;
                
                // Boundary check
                if (nx < 0 || nx >= resolution[0] ||
                    ny < 0 || ny >= resolution[1] ||
                    nz < 0 || nz >= resolution[2]) {
                    continue;
                }
                
                // Calculate neighbor cell hash
                int cell_hash = nz * resolution[1] * resolution[0] +
                                ny * resolution[0] + nx;
                
                int start = cell_starts[cell_hash];
                int end = cell_ends[cell_hash];
                
                // Empty cell check
                if (start < 0 || end < 0) continue;
                
                // Iterate through all objects in this cell
                for (int j = start; j < end; j++) {
                    // Avoid duplicate detection and self-collision
                    if (j <= idx) continue;
                    
                    // Read other object info
                    float qx = positions[j * 3 + 0];
                    float qy = positions[j * 3 + 1];
                    float qz = positions[j * 3 + 2];
                    float r2 = radii[j];
                    
                    // Sphere intersection test
                    float dx_val = px - qx;
                    float dy_val = py - qy;
                    float dz_val = pz - qz;
                    float dist_sq = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;
                    
                    float sum_r = r1 + r2;
                    
                    // If intersecting, record collision pair
                    if (dist_sq < sum_r * sum_r && dist_sq > 1e-10f) {
                        int pair_idx = atomicAdd(pair_count, 1);
                        if (pair_idx < max_pairs) {
                            collision_pairs[pair_idx * 2 + 0] = idx;
                            collision_pairs[pair_idx * 2 + 1] = j;
                        }
                    }
                }
            }
        }
    }
}
''', 'broad_phase_collision')


# ============================================================================
# 碰撞响应相关Kernel
# ============================================================================

COLLISION_RESPONSE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void resolve_collisions(
    float* positions,               // [N, 3] object positions (read-write)
    float* velocities,              // [N, 3] object velocities (read-write)
    const float* radii,             // [N] object radii
    const float* masses,            // [N] object masses
    const float* restitutions,      // [N] restitution coefficients
    const int* collision_pairs,     // [num_pairs, 2] collision pairs
    int num_pairs                   // number of collision pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    // Read collision pair IDs
    int id_a = collision_pairs[idx * 2 + 0];
    int id_b = collision_pairs[idx * 2 + 1];
    
    // Read object A data
    float px_a = positions[id_a * 3 + 0];
    float py_a = positions[id_a * 3 + 1];
    float pz_a = positions[id_a * 3 + 2];
    
    float vx_a = velocities[id_a * 3 + 0];
    float vy_a = velocities[id_a * 3 + 1];
    float vz_a = velocities[id_a * 3 + 2];
    
    float r_a = radii[id_a];
    float m_a = masses[id_a];
    float e_a = restitutions[id_a];
    
    // Read object B data
    float px_b = positions[id_b * 3 + 0];
    float py_b = positions[id_b * 3 + 1];
    float pz_b = positions[id_b * 3 + 2];
    
    float vx_b = velocities[id_b * 3 + 0];
    float vy_b = velocities[id_b * 3 + 1];
    float vz_b = velocities[id_b * 3 + 2];
    
    float r_b = radii[id_b];
    float m_b = masses[id_b];
    float e_b = restitutions[id_b];
    
    // Calculate collision normal
    float dx = px_b - px_a;
    float dy = py_b - py_a;
    float dz = pz_b - pz_a;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    
    // Avoid division by zero
    if (dist < 1e-6f) return;
    
    // Normalize normal
    float nx = dx / dist;
    float ny = dy / dist;
    float nz = dz / dist;
    
    // Calculate relative velocity
    float dvx = vx_b - vx_a;
    float dvy = vy_b - vy_a;
    float dvz = vz_b - vz_a;
    
    // Relative velocity along normal
    float vel_along_normal = dvx * nx + dvy * ny + dvz * nz;
    
    // Calculate penetration depth
    float penetration = (r_a + r_b) - dist;
    
    // Use minimum restitution coefficient
    float e = fminf(e_a, e_b);
    
    // Calculate impulse scalar (elastic collision)
    float j = 0.0f;
    if (vel_along_normal < 0) {
        // Objects approaching - apply normal impulse
        j = -(1.0f + e) * vel_along_normal;
        j /= (1.0f / m_a + 1.0f / m_b);
    } else if (penetration > 0.001f && fabsf(vel_along_normal) < 0.01f) {
        // Objects penetrating but not separating fast enough
        // Apply a correction impulse based on penetration depth
        // This prevents objects from "sticking" together
        float correction_velocity = penetration * 50.0f;  // Convert penetration to velocity
        j = correction_velocity;
        j /= (1.0f / m_a + 1.0f / m_b);
    }
    
    // Calculate impulse vector
    float jx = j * nx;
    float jy = j * ny;
    float jz = j * nz;
    
    // Apply impulse to velocities (use atomic ops to avoid race conditions)
    atomicAdd(&velocities[id_a * 3 + 0], -jx / m_a);
    atomicAdd(&velocities[id_a * 3 + 1], -jy / m_a);
    atomicAdd(&velocities[id_a * 3 + 2], -jz / m_a);
    
    atomicAdd(&velocities[id_b * 3 + 0], jx / m_b);
    atomicAdd(&velocities[id_b * 3 + 1], jy / m_b);
    atomicAdd(&velocities[id_b * 3 + 2], jz / m_b);
    
    // Position correction (avoid penetration)
    // Use stronger correction factor (0.9) for faster separation
    if (penetration > 0) {
        // Distribute correction proportional to mass
        float total_inv_mass = 1.0f / m_a + 1.0f / m_b;
        float correction_a = penetration * (1.0f / m_a) / total_inv_mass * 0.9f;
        float correction_b = penetration * (1.0f / m_b) / total_inv_mass * 0.9f;
        
        atomicAdd(&positions[id_a * 3 + 0], -nx * correction_a);
        atomicAdd(&positions[id_a * 3 + 1], -ny * correction_a);
        atomicAdd(&positions[id_a * 3 + 2], -nz * correction_a);
        
        atomicAdd(&positions[id_b * 3 + 0], nx * correction_b);
        atomicAdd(&positions[id_b * 3 + 1], ny * correction_b);
        atomicAdd(&positions[id_b * 3 + 2], nz * correction_b);
    }
}
''', 'resolve_collisions')


# ============================================================================
# 物理积分相关Kernel
# ============================================================================

INTEGRATE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void integrate_physics(
    float* positions,               // [N, 3] object positions (read-write)
    float* velocities,              // [N, 3] object velocities (read-write)
    const float* forces,            // [N, 3] accumulated forces
    const float* masses,            // [N] object masses
    const float* gravity,           // [3] gravity acceleration
    float dt,                       // time step
    float damping,                  // damping coefficient
    const float* bounds_min,        // [3] boundary minimum
    const float* bounds_max,        // [3] boundary maximum
    const float* radii,             // [N] object radii
    const float* restitutions,      // [N] restitution coefficients
    int num_objects                 // number of objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    float mass = masses[idx];
    float inv_mass = 1.0f / mass;
    float radius = radii[idx];
    float restitution = restitutions[idx];
    
    // Calculate total acceleration = force/mass + gravity
    float ax = forces[idx * 3 + 0] * inv_mass + gravity[0];
    float ay = forces[idx * 3 + 1] * inv_mass + gravity[1];
    float az = forces[idx * 3 + 2] * inv_mass + gravity[2];
    
    // Semi-implicit Euler integration: update velocity first
    float vx = velocities[idx * 3 + 0] + ax * dt;
    float vy = velocities[idx * 3 + 1] + ay * dt;
    float vz = velocities[idx * 3 + 2] + az * dt;
    
    // Apply damping
    float damping_factor = 1.0f - damping * dt;
    vx *= damping_factor;
    vy *= damping_factor;
    vz *= damping_factor;
    
    // Update position using new velocity
    float px = positions[idx * 3 + 0] + vx * dt;
    float py = positions[idx * 3 + 1] + vy * dt;
    float pz = positions[idx * 3 + 2] + vz * dt;
    
    // Boundary collision handling
    // X-axis boundaries
    if (px - radius < bounds_min[0]) {
        px = bounds_min[0] + radius;
        vx = -vx * restitution;
    } else if (px + radius > bounds_max[0]) {
        px = bounds_max[0] - radius;
        vx = -vx * restitution;
    }
    
    // Y-axis boundaries
    if (py - radius < bounds_min[1]) {
        py = bounds_min[1] + radius;
        vy = -vy * restitution;
    } else if (py + radius > bounds_max[1]) {
        py = bounds_max[1] - radius;
        vy = -vy * restitution;
    }
    
    // Z-axis boundaries
    if (pz - radius < bounds_min[2]) {
        pz = bounds_min[2] + radius;
        vz = -vz * restitution;
    } else if (pz + radius > bounds_max[2]) {
        pz = bounds_max[2] - radius;
        vz = -vz * restitution;
    }
    
    // Write back results
    positions[idx * 3 + 0] = px;
    positions[idx * 3 + 1] = py;
    positions[idx * 3 + 2] = pz;
    
    velocities[idx * 3 + 0] = vx;
    velocities[idx * 3 + 1] = vy;
    velocities[idx * 3 + 2] = vz;
}
''', 'integrate_physics')
