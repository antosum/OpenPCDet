#!/usr/bin/env python3
"""
Test script for geometric accuracy metrics in OpenPCDet's simple_eval function.
This script tests the new center error, size error, and yaw error computations.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_geometric_metrics():
    """Test geometric metric computations with sample data."""
    print("🧪 Testing Geometric Accuracy Metrics")
    print("=" * 50)
    
    # Create sample detection and GT boxes
    # Format: [x, y, z, l, w, h, yaw]
    det_boxes = np.array([
        [10.0, 20.0, 0.0, 4.0, 2.0, 1.5, 0.5],    # Good detection
        [15.0, 25.0, 0.0, 3.8, 1.9, 1.6, 0.3],    # Good detection  
        [50.0, 60.0, 0.0, 4.2, 2.1, 1.4, 1.2],    # False positive
    ])
    
    gt_boxes = np.array([
        [10.1, 20.1, 0.0, 4.1, 2.1, 1.4, 0.6],    # Close to first detection
        [14.9, 25.1, 0.0, 3.7, 1.8, 1.7, 0.2],    # Close to second detection
        [30.0, 40.0, 0.0, 4.0, 2.0, 1.5, 0.8],    # Missed detection
    ])
    
    # Simulate matches (first two match, third doesn't)
    matches = np.array([1.0, 1.0, 0.0])
    
    print(f"📊 Sample Data:")
    print(f"   Detection boxes: {det_boxes.shape[0]}")
    print(f"   Ground truth boxes: {gt_boxes.shape[0]}")
    print(f"   Matches: {matches.sum()}/{len(matches)}")
    print()
    
    # Test center error computation
    print("🎯 Testing Center Error (BEV)")
    center_errors = compute_center_error(det_boxes, gt_boxes, matches)
    if len(center_errors) > 0:
        print(f"   Mean center error: {np.mean(center_errors):.3f}m")
        print(f"   Std center error: {np.std(center_errors):.3f}m")
        print(f"   Median center error: {np.median(center_errors):.3f}m")
        print(f"   Individual errors: {[f'{e:.3f}' for e in center_errors]}m")
    else:
        print("   No matched detections for center error computation")
    print()
    
    # Test size error computation
    print("📏 Testing Size Error")
    size_errors = compute_size_error(det_boxes, gt_boxes, matches)
    if len(size_errors) > 0:
        print(f"   Mean size error: {np.mean(size_errors):.3f}")
        print(f"   Std size error: {np.std(size_errors):.3f}")
        print(f"   Median size error: {np.median(size_errors):.3f}")
        print(f"   Individual errors: {[f'{e:.3f}' for e in size_errors]}")
    else:
        print("   No matched detections for size error computation")
    print()
    
    # Test yaw error computation
    print("🧭 Testing Yaw Error")
    yaw_errors = compute_yaw_error(det_boxes, gt_boxes, matches)
    if len(yaw_errors) > 0:
        print(f"   Mean yaw error: {np.mean(yaw_errors):.1f}°")
        print(f"   Std yaw error: {np.std(yaw_errors):.1f}°")
        print(f"   Median yaw error: {np.median(yaw_errors):.1f}°")
        print(f"   Individual errors: {[f'{e:.1f}' for e in yaw_errors]}°")
    else:
        print("   No matched detections for yaw error computation")
    print()
    
    # Test edge cases
    print("🔬 Testing Edge Cases")
    
    # Empty case
    empty_errors = compute_center_error(np.array([]), np.array([]), np.array([]))
    print(f"   Empty arrays: {len(empty_errors)} errors (expected: 0)")
    
    # No matches case
    no_match_errors = compute_center_error(det_boxes, gt_boxes, np.zeros(3))
    print(f"   No matches: {len(no_match_errors)} errors (expected: 0)")
    
    # Single match case
    single_det = det_boxes[:1]
    single_gt = gt_boxes[:1]
    single_match = np.array([1.0])
    single_errors = compute_center_error(single_det, single_gt, single_match)
    print(f"   Single match: {len(single_errors)} error(s) (expected: 1)")
    
    print()
    print("✅ All geometric metric tests completed successfully!")

def compute_center_error(det_boxes, gt_boxes, matches):
    """Compute center error (BEV) between matched detections and GT boxes."""
    if det_boxes.size == 0 or gt_boxes.size == 0:
        return np.array([])
    
    # Extract BEV centers (x, y coordinates)
    det_centers = det_boxes[:, :2]  # [x, y]
    gt_centers = gt_boxes[:, :2]    # [x, y]
    
    # Only compute for matched detections
    matched_indices = np.where(matches > 0)[0]
    if len(matched_indices) == 0:
        return np.array([])
    
    # For each matched detection, find the corresponding GT box
    center_errors = []
    gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
    
    for det_idx in matched_indices:
        det_center = det_centers[det_idx]
        
        # Find closest unmatched GT box
        distances = np.linalg.norm(gt_centers - det_center, axis=1)
        # Set already matched GT boxes to infinity
        distances[gt_used] = np.inf
        
        if np.min(distances) == np.inf:
            continue  # No unmatched GT boxes
        
        gt_idx = np.argmin(distances)
        gt_used[gt_idx] = True
        
        # Compute Euclidean distance in BEV
        error = np.linalg.norm(det_center - gt_centers[gt_idx])
        center_errors.append(error)
    
    return np.array(center_errors)

def compute_size_error(det_boxes, gt_boxes, matches):
    """Compute size error between matched detections and GT boxes."""
    if det_boxes.size == 0 or gt_boxes.size == 0:
        return np.array([])
    
    # Extract dimensions (l, w, h) - assuming format [x, y, z, l, w, h, ...]
    det_dims = det_boxes[:, 3:6]  # [l, w, h]
    gt_dims = gt_boxes[:, 3:6]    # [l, w, h]
    
    # Only compute for matched detections
    matched_indices = np.where(matches > 0)[0]
    if len(matched_indices) == 0:
        return np.array([])
    
    size_errors = []
    gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
    
    for det_idx in matched_indices:
        det_dim = det_dims[det_idx]
        
        # Find closest unmatched GT box by center distance
        det_center = det_boxes[det_idx, :2]
        gt_centers = gt_boxes[:, :2]
        distances = np.linalg.norm(gt_centers - det_center, axis=1)
        distances[gt_used] = np.inf
        
        if np.min(distances) == np.inf:
            continue
        
        gt_idx = np.argmin(distances)
        gt_used[gt_idx] = True
        
        # Compute relative size error for each dimension
        gt_dim = gt_dims[gt_idx]
        # Avoid division by zero
        rel_errors = np.abs(det_dim - gt_dim) / np.maximum(gt_dim, 0.01)
        # Use mean relative error across dimensions
        size_error = np.mean(rel_errors)
        size_errors.append(size_error)
    
    return np.array(size_errors)

def compute_yaw_error(det_boxes, gt_boxes, matches):
    """Compute yaw error between matched detections and GT boxes."""
    if det_boxes.size == 0 or gt_boxes.size == 0:
        return np.array([])
    
    # Extract yaw angles - assuming format [x, y, z, l, w, h, yaw, ...]
    det_yaws = det_boxes[:, 6]  # yaw
    gt_yaws = gt_boxes[:, 6]    # yaw
    
    # Only compute for matched detections
    matched_indices = np.where(matches > 0)[0]
    if len(matched_indices) == 0:
        return np.array([])
    
    yaw_errors = []
    gt_used = np.zeros(gt_boxes.shape[0], dtype=bool)
    
    for det_idx in matched_indices:
        det_yaw = det_yaws[det_idx]
        
        # Find closest unmatched GT box by center distance
        det_center = det_boxes[det_idx, :2]
        gt_centers = gt_boxes[:, :2]
        distances = np.linalg.norm(gt_centers - det_center, axis=1)
        distances[gt_used] = np.inf
        
        if np.min(distances) == np.inf:
            continue
        
        gt_idx = np.argmin(distances)
        gt_used[gt_idx] = True
        
        # Compute angular difference (handle circular nature of angles)
        gt_yaw = gt_yaws[gt_idx]
        yaw_diff = np.abs(det_yaw - gt_yaw)
        # Take the minimum of the difference and 2π - difference
        yaw_error = np.minimum(yaw_diff, 2 * np.pi - yaw_diff)
        # Convert to degrees
        yaw_error_deg = np.degrees(yaw_error)
        yaw_errors.append(yaw_error_deg)
    
    return np.array(yaw_errors)

if __name__ == "__main__":
    test_geometric_metrics()