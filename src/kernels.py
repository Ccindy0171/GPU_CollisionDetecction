"""
CUDA Kernel Functions Module

This module contains all GPU-accelerated CUDA kernel functions for collision detection
and physics simulation. The kernels are written in CUDA C and compiled at runtime
using CuPy's RawKernel interface.

Performance Characteristics:
    - Optimized for modern NVIDIA GPUs (tested on RTX 3050: 4GB VRAM, 2048 CUDA cores)
    - Uses coalesced memory access patterns for optimal bandwidth
    - Employs atomic operations where necessary to handle race conditions
    - Block size of 256 threads provides good occupancy on most GPUs

Memory Layout:
    - All data uses Structure of Arrays (SOA) for efficient GPU memory access
    - Positions, velocities stored as [N, 3] flat arrays (N objects, 3 components)
    - Scalars (radii, masses) stored as [N] flat arrays

Kernel Functions:
    COMPUTE_GRID_HASH_KERNEL: Maps 3D positions to 1D spatial grid indices
    FIND_CELL_START_KERNEL: Identifies start/end positions of objects in each grid cell
    BROAD_PHASE_KERNEL: Detects potential collision pairs using spatial grid
    COLLISION_RESPONSE_KERNEL: Resolves collisions using impulse-based physics
    INTEGRATE_KERNEL: Updates positions and velocities using semi-implicit Euler method

References:
    - "GPU Gems 3: Chapter 32" - Broad-Phase Collision Detection with CUDA
    - "Real-Time Collision Detection" by Christer Ericson
    - CuPy RawKernel documentation: https://docs.cupy.dev/en/stable/user_guide/kernel.html
"""

import cupy as cp


# ============================================================================
# Grid Construction Kernels
# ============================================================================

COMPUTE_GRID_HASH_KERNEL = cp.RawKernel(r'''
/**
 * Compute Grid Hash Kernel
 * 
 * Purpose: Maps each object's 3D position to a 1D grid cell hash value.
 * This is the first step in building the spatial acceleration structure.
 * 
 * Algorithm:
 *   1. Convert world position to grid coordinates by dividing by cell size
 *   2. Clamp coordinates to valid grid bounds
 *   3. Convert 3D grid coordinates to 1D hash using row-major ordering
 * 
 * Performance: O(1) per object, fully parallel
 * Thread mapping: One thread per object
 * 
 * Memory access pattern: Coalesced reads from positions array
 */
extern "C" __global__
void compute_grid_hash(
    const float* positions,     // [N, 3] object positions in world space
    int* grid_hashes,           // [N] output: 1D grid cell hash for each object
    const float* world_min,     // [3] world space minimum bounds (origin)
    float cell_size,            // spatial size of each grid cell
    const int* resolution,      // [3] grid dimensions (nx, ny, nz)
    int num_objects             // total number of objects
) {
    // Calculate global thread ID (one thread per object)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // Read object position from global memory (coalesced access)
    float px = positions[idx * 3 + 0];  // x coordinate
    float py = positions[idx * 3 + 1];  // y coordinate
    float pz = positions[idx * 3 + 2];  // z coordinate
    
    // Transform world coordinates to grid coordinates
    // Grid origin is at world_min, each cell has size cell_size
    int gx = (int)((px - world_min[0]) / cell_size);
    int gy = (int)((py - world_min[1]) / cell_size);
    int gz = (int)((pz - world_min[2]) / cell_size);
    
    // Clamp grid coordinates to valid bounds [0, resolution-1]
    // This handles objects slightly outside the world bounds
    gx = max(0, min(gx, resolution[0] - 1));
    gy = max(0, min(gy, resolution[1] - 1));
    gz = max(0, min(gz, resolution[2] - 1));
    
    // Convert 3D grid coordinates to 1D hash (row-major order: z*ny*nx + y*nx + x)
    // This spatial hash ensures nearby objects have similar hash values
    grid_hashes[idx] = gz * resolution[1] * resolution[0] + 
                       gy * resolution[0] + gx;
}
''', 'compute_grid_hash')


FIND_CELL_START_KERNEL = cp.RawKernel(r'''
/**
 * Find Cell Start Kernel
 * 
 * Purpose: After sorting objects by grid hash, identifies the start and end
 * indices for each grid cell. This enables rapid lookup of objects in a cell.
 * 
 * Algorithm:
 *   For each sorted object:
 *   - If it's the first object in its cell (hash differs from previous), mark cell start
 *   - If it's the last object in its cell (hash differs from next), mark cell end
 * 
 * Prerequisites: Object indices must be sorted by grid hash
 * 
 * Performance: O(1) per object, fully parallel
 * Thread mapping: One thread per object
 * 
 * Note: Uses atomic-free algorithm - each thread only writes to unique locations
 */
extern "C" __global__
void find_cell_start(
    const int* sorted_hashes,   // [N] sorted grid hash values (ascending order)
    int* cell_starts,           // [total_cells] output: starting index for each cell
    int* cell_ends,             // [total_cells] output: ending index (exclusive) for each cell
    int num_objects             // total number of objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    int hash = sorted_hashes[idx];
    
    // Check if this is the first element in the cell
    // Either idx==0 (first overall) or previous hash is different
    if (idx == 0 || sorted_hashes[idx - 1] != hash) {
        cell_starts[hash] = idx;
    }
    
    // Check if this is the last element in the cell
    // Either idx==N-1 (last overall) or next hash is different
    if (idx == num_objects - 1 || sorted_hashes[idx + 1] != hash) {
        cell_ends[hash] = idx + 1;  // end index is exclusive (Python-style)
    }
}
''', 'find_cell_start')


# ============================================================================
# Collision Detection Kernels
# ============================================================================

BROAD_PHASE_KERNEL = cp.RawKernel(r'''
/**
 * Broad Phase Collision Detection Kernel
 * 
 * Purpose: Efficiently detect all potential collision pairs using spatial grid.
 * This is the "broad phase" that quickly filters out distant object pairs.
 * 
 * Algorithm:
 *   For each object:
 *   1. Determine which grid cell it occupies
 *   2. Check all 27 neighboring cells (3x3x3 neighborhood)
 *   3. Test sphere-sphere intersection with objects in those cells
 *   4. Record colliding pairs using atomic counter
 * 
 * Why 27 cells? Objects near cell boundaries might collide with objects in
 * adjacent cells. Checking 3x3x3 = 27 cells ensures no collisions are missed.
 * 
 * Performance: O(k) per object where k = avg objects per cell × 27
 * Thread mapping: One thread per object
 * 
 * Optimization: Using spatial grid reduces from O(N²) to O(N×k) complexity
 * 
 * Memory access: Coalesced reads, atomic writes for collision pairs
 */
extern "C" __global__
void broad_phase_collision(
    const float* positions,         // [N, 3] object positions in world space
    const float* radii,             // [N] object radii (sphere collision detection)
    const int* cell_starts,         // [total_cells] starting index for each grid cell
    const int* cell_ends,           // [total_cells] ending index for each grid cell
    const int* sorted_indices,      // [N] object indices sorted by grid hash
    const int* resolution,          // [3] grid resolution (nx, ny, nz)
    float cell_size,                // spatial size of each grid cell
    const float* world_min,         // [3] world space minimum bounds
    int* collision_pairs,           // [max_pairs, 2] output: pairs of colliding object indices
    int* pair_count,                // [1] output: total number of collision pairs (atomic counter)
    int num_objects,                // total number of objects
    int max_pairs                   // maximum collision pairs buffer can hold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // Read current object properties from global memory
    float px = positions[idx * 3 + 0];  // position x
    float py = positions[idx * 3 + 1];  // position y
    float pz = positions[idx * 3 + 2];  // position z
    float r1 = radii[idx];              // radius
    
    // Calculate which grid cell this object occupies
    int gx = (int)((px - world_min[0]) / cell_size);
    int gy = (int)((py - world_min[1]) / cell_size);
    int gz = (int)((pz - world_min[2]) / cell_size);
    
    // Iterate through 27 neighboring cells (3x3x3 neighborhood)
    // This ensures we don't miss collisions with objects in adjacent cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                // Calculate neighbor cell coordinates
                int nx = gx + dx;
                int ny = gy + dy;
                int nz = gz + dz;
                
                // Boundary check: skip cells outside the grid
                if (nx < 0 || nx >= resolution[0] ||
                    ny < 0 || ny >= resolution[1] ||
                    nz < 0 || nz >= resolution[2]) {
                    continue;
                }
                
                // Calculate neighbor cell's 1D hash
                int cell_hash = nz * resolution[1] * resolution[0] +
                                ny * resolution[0] + nx;
                
                // Get the range of objects in this cell
                int start = cell_starts[cell_hash];
                int end = cell_ends[cell_hash];
                
                // Empty cell check: skip if no objects in this cell
                if (start < 0 || end < 0) continue;
                
                // Iterate through all objects in this neighbor cell
                for (int sorted_j = start; sorted_j < end; sorted_j++) {
                    // Get the actual object index from sorted array
                    int j = sorted_indices[sorted_j];
                    
                    // Avoid duplicate detection (only check pairs once: i < j)
                    // Also prevents self-collision (i != i)
                    if (j <= idx) continue;
                    
                    // Read other object properties
                    float qx = positions[j * 3 + 0];
                    float qy = positions[j * 3 + 1];
                    float qz = positions[j * 3 + 2];
                    float r2 = radii[j];
                    
                    // Sphere intersection test: check if distance < sum of radii
                    float dx_val = px - qx;
                    float dy_val = py - qy;
                    float dz_val = pz - qz;
                    float dist_sq = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;
                    
                    float sum_r = r1 + r2;
                    
                    // If spheres intersect, record collision pair
                    // dist_sq > 1e-10 avoids numerical issues when objects exactly overlap
                    if (dist_sq < sum_r * sum_r && dist_sq > 1e-10f) {
                        // Atomically increment pair counter and get index
                        int pair_idx = atomicAdd(pair_count, 1);
                        
                        // Only write if buffer has space (prevent overflow)
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
# Collision Response Kernels
# ============================================================================

COLLISION_RESPONSE_KERNEL = cp.RawKernel(r'''
/**
 * Collision Response Kernel
 * 
 * Purpose: Resolves detected collisions by applying impulse-based physics.
 * Updates velocities and positions to separate overlapping objects.
 * 
 * Physics Model: Impulse-based collision response
 *   - Calculates collision normal (direction from A to B)
 *   - Computes relative velocity along normal
 *   - Applies impulse based on coefficient of restitution (elasticity)
 *   - Separates penetrating objects proportional to their masses
 * 
 * Restitution coefficient (e):
 *   e = 0.0: Perfectly inelastic (objects stick together)
 *   e = 1.0: Perfectly elastic (objects bounce with no energy loss)
 *   e ∈ (0,1): Partially elastic (realistic materials)
 * 
 * Algorithm:
 *   1. Calculate collision normal and penetration depth
 *   2. Compute relative velocity along normal
 *   3. Apply velocity impulse if objects are approaching
 *   4. Apply position correction to resolve penetration
 * 
 * Performance: O(1) per collision pair, fully parallel
 * Thread mapping: One thread per collision pair
 * 
 * Thread Safety: Uses atomic operations to safely update shared object data
 * 
 * References:
 *   - "Game Physics Engine Development" by Ian Millington (Chapter 14)
 *   - "Real-Time Rendering" (Chapter on collision response)
 */
extern "C" __global__
void resolve_collisions(
    float* positions,               // [N, 3] object positions (read-write)
    float* velocities,              // [N, 3] object velocities (read-write)
    const float* radii,             // [N] object radii
    const float* masses,            // [N] object masses
    const float* restitutions,      // [N] restitution coefficients (elasticity)
    const int* collision_pairs,     // [num_pairs, 2] pairs of colliding object indices
    int num_pairs                   // number of collision pairs to process
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    // Read collision pair object IDs
    int id_a = collision_pairs[idx * 2 + 0];
    int id_b = collision_pairs[idx * 2 + 1];
    
    // Read object A properties from global memory
    float px_a = positions[id_a * 3 + 0];
    float py_a = positions[id_a * 3 + 1];
    float pz_a = positions[id_a * 3 + 2];
    
    float vx_a = velocities[id_a * 3 + 0];
    float vy_a = velocities[id_a * 3 + 1];
    float vz_a = velocities[id_a * 3 + 2];
    
    float r_a = radii[id_a];
    float m_a = masses[id_a];
    float e_a = restitutions[id_a];
    
    // Read object B properties from global memory
    float px_b = positions[id_b * 3 + 0];
    float py_b = positions[id_b * 3 + 1];
    float pz_b = positions[id_b * 3 + 2];
    
    float vx_b = velocities[id_b * 3 + 0];
    float vy_b = velocities[id_b * 3 + 1];
    float vz_b = velocities[id_b * 3 + 2];
    
    float r_b = radii[id_b];
    float m_b = masses[id_b];
    float e_b = restitutions[id_b];
    
    // Calculate collision normal: direction from A to B
    float dx = px_b - px_a;
    float dy = py_b - py_a;
    float dz = pz_b - pz_a;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    
    // Avoid division by zero (objects at exact same position)
    if (dist < 1e-6f) return;
    
    // Normalize collision normal to unit vector
    float nx = dx / dist;
    float ny = dy / dist;
    float nz = dz / dist;
    
    // Calculate relative velocity (B velocity minus A velocity)
    float dvx = vx_b - vx_a;
    float dvy = vy_b - vy_a;
    float dvz = vz_b - vz_a;
    
    // Project relative velocity onto collision normal
    // Positive = objects separating, Negative = objects approaching
    float vel_along_normal = dvx * nx + dvy * ny + dvz * nz;
    
    // Calculate penetration depth (how much spheres overlap)
    float penetration = (r_a + r_b) - dist;
    
    // Use minimum restitution coefficient (more conservative)
    // This prevents unrealistic energy gain in collisions
    float e = fminf(e_a, e_b);
    
    // Calculate impulse scalar using impulse-momentum theorem
    // j = -(1 + e) * v_rel · n / (1/m_a + 1/m_b)
    float j = 0.0f;
    if (vel_along_normal < 0) {
        // Objects are approaching - apply normal collision impulse
        // The (1 + e) factor controls how much energy is conserved
        j = -(1.0f + e) * vel_along_normal;
        j /= (1.0f / m_a + 1.0f / m_b);
    } else if (penetration > 0.001f && fabsf(vel_along_normal) < 0.01f) {
        // Objects are penetrating but not separating fast enough
        // Apply correction impulse to prevent "sticking" behavior
        // Convert penetration depth to velocity (penetration * stiffness)
        float correction_velocity = penetration * 50.0f;  // stiffness factor
        j = correction_velocity;
        j /= (1.0f / m_a + 1.0f / m_b);
    }
    
    // Convert impulse scalar to vector (along collision normal)
    float jx = j * nx;
    float jy = j * ny;
    float jz = j * nz;
    
    // Apply impulse to velocities using Newton's second law: Δv = j/m
    // Use atomic operations to safely handle multiple threads modifying same object
    // Object A receives negative impulse (pushed away from B)
    atomicAdd(&velocities[id_a * 3 + 0], -jx / m_a);
    atomicAdd(&velocities[id_a * 3 + 1], -jy / m_a);
    atomicAdd(&velocities[id_a * 3 + 2], -jz / m_a);
    
    // Object B receives positive impulse (pushed away from A)
    atomicAdd(&velocities[id_b * 3 + 0], jx / m_b);
    atomicAdd(&velocities[id_b * 3 + 1], jy / m_b);
    atomicAdd(&velocities[id_b * 3 + 2], jz / m_b);
    
    // Position correction to resolve penetration (prevents "sinking")
    // Uses Baumgarte stabilization with factor 0.9 for fast separation
    if (penetration > 0) {
        // Distribute correction inversely proportional to mass
        // Heavier objects move less, lighter objects move more
        float total_inv_mass = 1.0f / m_a + 1.0f / m_b;
        float correction_a = penetration * (1.0f / m_a) / total_inv_mass * 0.9f;
        float correction_b = penetration * (1.0f / m_b) / total_inv_mass * 0.9f;
        
        // Apply position corrections using atomic operations
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
# Physics Integration Kernel
# ============================================================================

INTEGRATE_KERNEL = cp.RawKernel(r'''
/**
 * Physics Integration Kernel
 * 
 * Purpose: Updates object positions and velocities based on forces, and handles
 * collisions with world boundaries.
 * 
 * Integration Method: Semi-implicit (Symplectic) Euler
 *   1. Update velocity: v(t+dt) = v(t) + a(t) * dt
 *   2. Update position: p(t+dt) = p(t) + v(t+dt) * dt
 * 
 * Why semi-implicit? More stable than explicit Euler for physics simulation.
 * It conserves energy better and prevents instabilities at larger timesteps.
 * 
 * Features:
 *   - Applies external forces (gravity, user forces)
 *   - Applies velocity damping (air resistance)
 *   - Handles boundary collisions with coefficient of restitution
 *   - Ensures objects stay within world bounds
 * 
 * Boundary Handling:
 *   When object hits a boundary:
 *   - Clamp position to boundary (with radius offset)
 *   - Reverse and scale velocity by restitution coefficient
 *   - This simulates elastic/inelastic bouncing
 * 
 * Performance: O(1) per object, fully parallel
 * Thread mapping: One thread per object
 * 
 * References:
 *   - "Game Physics Pearls" (Chapter on numerical integration)
 *   - "Foundations of Physically Based Modeling" - Witkin & Baraff
 */
extern "C" __global__
void integrate_physics(
    float* positions,               // [N, 3] object positions (read-write)
    float* velocities,              // [N, 3] object velocities (read-write)
    const float* forces,            // [N, 3] accumulated external forces
    const float* masses,            // [N] object masses
    const float* gravity,           // [3] global gravity acceleration (e.g., [0, -9.81, 0])
    float dt,                       // time step (seconds, typically 1/60)
    float damping,                  // velocity damping coefficient [0,1] (air resistance)
    const float* bounds_min,        // [3] world boundary minimum (x, y, z)
    const float* bounds_max,        // [3] world boundary maximum (x, y, z)
    const float* radii,             // [N] object radii
    const float* restitutions,      // [N] restitution coefficients for boundary collisions
    int num_objects                 // total number of objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_objects) return;
    
    // Read object properties
    float mass = masses[idx];
    float inv_mass = 1.0f / mass;        // Inverse mass for efficiency
    float radius = radii[idx];
    float restitution = restitutions[idx];
    
    // Calculate total acceleration: F/m + gravity
    // Newton's second law: a = F/m
    float ax = forces[idx * 3 + 0] * inv_mass + gravity[0];
    float ay = forces[idx * 3 + 1] * inv_mass + gravity[1];
    float az = forces[idx * 3 + 2] * inv_mass + gravity[2];
    
    // Semi-implicit Euler integration: update velocity first
    // v(t+dt) = v(t) + a(t) * dt
    float vx = velocities[idx * 3 + 0] + ax * dt;
    float vy = velocities[idx * 3 + 1] + ay * dt;
    float vz = velocities[idx * 3 + 2] + az * dt;
    
    // Apply damping to simulate air resistance
    // v *= (1 - damping * dt)
    // damping = 0: no damping, damping = 1: full damping
    float damping_factor = 1.0f - damping * dt;
    vx *= damping_factor;
    vy *= damping_factor;
    vz *= damping_factor;
    
    // Update position using new velocity (symplectic step)
    // p(t+dt) = p(t) + v(t+dt) * dt
    float px = positions[idx * 3 + 0] + vx * dt;
    float py = positions[idx * 3 + 1] + vy * dt;
    float pz = positions[idx * 3 + 2] + vz * dt;
    
    // Boundary collision handling for all three axes
    // X-axis boundaries
    if (px - radius < bounds_min[0]) {
        // Hit minimum X boundary
        px = bounds_min[0] + radius;      // Clamp position
        vx = -vx * restitution;           // Reverse and dampen velocity
    } else if (px + radius > bounds_max[0]) {
        // Hit maximum X boundary
        px = bounds_max[0] - radius;
        vx = -vx * restitution;
    }
    
    // Y-axis boundaries (typically vertical, with floor at min)
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
    
    // Write updated positions and velocities back to global memory
    positions[idx * 3 + 0] = px;
    positions[idx * 3 + 1] = py;
    positions[idx * 3 + 2] = pz;
    
    velocities[idx * 3 + 0] = vx;
    velocities[idx * 3 + 1] = vy;
    velocities[idx * 3 + 2] = vz;
}
''', 'integrate_physics')
