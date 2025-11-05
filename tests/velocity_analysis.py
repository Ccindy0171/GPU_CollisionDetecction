#!/usr/bin/env python3
"""Check relative velocities in stacked balls"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

def analyze_relative_velocities():
    """Check relative velocity along normal for stacked balls"""
    print("=" * 70)
    print("RELATIVE VELOCITY ANALYSIS")
    print("=" * 70)
    
    # Simulate stacked balls all falling
    num_balls = 3
    positions = np.array([
        [0.0, 5.0, 0.0],
        [0.0, 5.5, 0.0],  # 0.5m above
        [0.0, 6.0, 0.0],  # 1.0m above
    ], dtype=np.float32)
    
    # All falling at same speed
    velocities = np.array([
        [0.0, -5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, -5.0, 0.0],
    ], dtype=np.float32)
    
    radii = np.full(num_balls, 0.3, dtype=np.float32)
    
    print("\n1. SETUP:")
    print(f"   All balls falling at -5.0 m/s in Y direction")
    print(f"   Ball radius: {radii[0]:.2f}m")
    print(f"   Spacing: 0.5m")
    print()
    
    print("2. PAIRWISE ANALYSIS:")
    for i in range(num_balls):
        for j in range(i+1, num_balls):
            # Calculate collision normal (from i to j)
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            dist = np.linalg.norm([dx, dy, dz])
            
            nx = dx / dist
            ny = dy / dist
            nz = dz / dist
            
            # Calculate relative velocity (j - i)
            dvx = velocities[j, 0] - velocities[i, 0]
            dvy = velocities[j, 1] - velocities[i, 1]
            dvz = velocities[j, 2] - velocities[i, 2]
            
            # Relative velocity along normal
            vel_along_normal = dvx * nx + dvy * ny + dvz * nz
            
            # Check if overlapping
            min_dist = radii[i] + radii[j]
            overlapping = dist < min_dist
            
            print(f"\n   Pair ({i}, {j}):")
            print(f"      Distance: {dist:.3f}m, Min dist: {min_dist:.3f}m")
            print(f"      Overlapping: {overlapping}")
            print(f"      Normal: ({nx:.3f}, {ny:.3f}, {nz:.3f})")
            print(f"      Relative velocity: ({dvx:.3f}, {dvy:.3f}, {dvz:.3f})")
            print(f"      Vel along normal: {vel_along_normal:.6f}")
            
            if vel_along_normal > 0:
                print(f"      → Separating (would skip impulse)")
            elif vel_along_normal < 0:
                print(f"      → Approaching (would apply impulse)")
            else:
                print(f"      → No relative motion (j = 0)")
    
    print("\n3. CONCLUSION:")
    print("   When all balls fall at same speed:")
    print("   - Relative velocity = 0")
    print("   - vel_along_normal = 0")
    print("   - Impulse j = 0")
    print("   - NO velocity change!")
    print()
    print("   This is why penetration_test shows identical velocities.")
    print("   Position correction works, but velocity doesn't change.")

if __name__ == "__main__":
    analyze_relative_velocities()
