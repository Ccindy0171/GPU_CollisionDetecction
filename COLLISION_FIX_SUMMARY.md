# Collision Detection Bug Fix - November 5, 2025

## Problem Description

The gravity fall simulation was experiencing severe tunneling issues where balls were passing through each other despite the collision detection system being active. With 10 balls, overlaps were occurring in 204 out of 300 frames (68%), and **zero collisions were being detected**.

## Root Cause

The bug was in the `BROAD_PHASE_KERNEL` in `src/kernels.py`. The kernel implements a spatial hash grid collision detection algorithm where:

1. Objects are sorted by their grid cell hash
2. Each cell stores start/end indices into the sorted array
3. For each object, we check its 27 neighboring cells for potential collisions

**The Bug:** When iterating through objects in a cell, the kernel was using the sorted array index `j` directly as an object ID to read from the `positions` array, instead of dereferencing through the `sorted_indices` array first.

```cuda
// WRONG - Before fix:
for (int j = start; j < end; j++) {
    if (j <= idx) continue;  // j is sorted index, but comparing with object ID!
    float qx = positions[j * 3 + 0];  // Reading wrong position!
    float qy = positions[j * 3 + 1];
    float qz = positions[j * 3 + 2];
    float r2 = radii[j];  // Reading wrong radius!
}

// CORRECT - After fix:
for (int sorted_j = start; sorted_j < end; sorted_j++) {
    int j = sorted_indices[sorted_j];  // Map sorted index to object ID
    if (j <= idx) continue;  // Now comparing object IDs correctly
    float qx = positions[j * 3 + 0];  // Reading correct position
    float qy = positions[j * 3 + 1];
    float qz = positions[j * 3 + 2];
    float r2 = radii[j];  // Reading correct radius
}
```

This caused the collision detection to read random/wrong positions and radii, resulting in no collisions being detected even when balls were clearly overlapping.

## Solution

### 1. Fixed the Kernel (src/kernels.py)

Added `sorted_indices` parameter to `BROAD_PHASE_KERNEL` and correctly dereferenced it when iterating through cell contents.

**Changes:**
- Added parameter: `const int* sorted_indices`
- Changed loop variable: `sorted_j` instead of `j`
- Added mapping: `int j = sorted_indices[sorted_j];`

### 2. Updated the Kernel Call (src/simulator.py)

Updated the `detect_collisions()` method to pass `self.sorted_indices` to the kernel.

**Changes:**
- Added `self.sorted_indices` as parameter to `BROAD_PHASE_KERNEL` call

### 3. Improved Configuration (examples/gravity_fall.py)

While debugging, also improved the default settings:
- Reduced `CELL_SIZE` from 2.0 to 1.0 for better spatial resolution with small balls
- Added comprehensive overlap detection and reporting
- Added minimum distance tracking for diagnostics

## Results

### Before Fix:
```
Testing with CELL_SIZE = 2.0
  Total collisions: 0
  Frames with overlaps: 204/300
  Max simultaneous overlaps: 1
  ✗✗✗ SIGNIFICANT TUNNELING!
```

### After Fix:
```
Testing with CELL_SIZE = 2.0
  Total collisions: 33
  Frames with overlaps: 0/300
  Max simultaneous overlaps: 0
  ✓✓✓ NO TUNNELING DETECTED!
```

## Impact

This was a **critical bug** that completely broke the collision detection system. The fix:
- ✅ Eliminates all tunneling/pass-through issues
- ✅ Enables proper collision detection (0 → 33+ collisions detected)
- ✅ Maintains physics accuracy (zero persistent overlaps)
- ✅ No performance impact (same algorithm, just correct implementation)

## Testing

Created `tests/test_tunneling_diagnostic.py` to validate collision detection across different cell sizes:
- Tests with 10 balls falling under gravity
- Runs 300 frames (5 seconds of simulation)
- Checks for overlaps at every frame
- Validates collision counts

All configurations now pass with **zero tunneling**.

## Files Modified

1. **src/kernels.py** - Fixed BROAD_PHASE_KERNEL sorted index bug
2. **src/simulator.py** - Updated kernel call to pass sorted_indices
3. **examples/gravity_fall.py** - Improved diagnostics and configuration
4. **tests/test_tunneling_diagnostic.py** - Added comprehensive tunneling test

## Lessons Learned

1. **Index Mapping**: When using spatial data structures with sorting, always maintain proper index mapping between sorted and original arrays.

2. **Diagnostic Tests**: The tunneling diagnostic test was crucial for identifying the issue - it showed zero collisions despite clear overlaps, pointing directly to the detection algorithm.

3. **Kernel Validation**: GPU kernels are harder to debug than CPU code - comprehensive diagnostic output is essential.

## Future Recommendations

1. Add assertions/validation in the kernel to detect when sorted vs. unsorted indices are mixed
2. Consider using structs or separate arrays for sorted vs. unsorted data to make the distinction clearer
3. Add unit tests specifically for the spatial hash grid to catch index mapping issues early
