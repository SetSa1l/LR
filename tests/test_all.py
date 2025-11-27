# -*- coding: utf-8 -*-
"""
Consolidated tests for Linear Region Analysis Toolkit

This module combines all tests from:
- test1.py (Phase 1: ModelWrapper and LinearRegionTraverser)
- test2.py (Phase 2: DecisionBoundaryDirectionFinder)
- test3.py (Phase 3: RegionPropertyAnalyzer)
- test4.py (Phase 4: LinearRegionAnalyzer)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my import (
    ModelWrapper,
    LinearRegionTraverser,
    normalize_direction,
    DecisionBoundaryDirectionFinder,
    find_decision_boundary_direction,
    RegionPropertyAnalyzer,
    LinearRegionAnalyzer
)
from my.core.model_wrapper import EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_simple_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10):
    """Create a simple MLP for testing"""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def create_simple_cnn(num_classes=10):
    """Create a simple CNN for testing"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, num_classes)
    )


def get_device():
    """Get available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Phase 1 Tests: ModelWrapper and LinearRegionTraverser
# =============================================================================

def test_direction_normalization():
    """Test 1.1: Direction Normalization"""
    print("=" * 50)
    print("Test 1.1: Direction Normalization")
    print("=" * 50)
    
    direction = torch.randn(4, 3, 32, 32)
    normalized = normalize_direction(direction)
    
    norms = torch.norm(normalized.view(4, -1), p=2, dim=1)
    print("Norms after normalization: %s" % norms)
    
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("PASSED\n")
    return True


def test_model_wrapper_mlp():
    """Test 1.2: ModelWrapper with MLP"""
    print("=" * 50)
    print("Test 1.2: ModelWrapper with MLP")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(784,), batch_size=32, device=device)
    
    print("Number of ReLU layers: %d" % wrapper.num_relus)
    assert wrapper.num_relus == 2
    
    x = torch.randn(4, 784, device=device)
    direction = normalize_direction(torch.randn(4, 784, device=device))
    
    state = wrapper.get_region_state(x, direction)
    
    print("Logits shape: %s" % str(state.logits.shape))
    print("Lambda to boundary: %s" % state.lambda_to_boundary)
    
    assert state.logits.shape == (4, 10)
    assert state.lambda_to_boundary.shape == (4,)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_model_wrapper_cnn():
    """Test 1.3: ModelWrapper with CNN"""
    print("=" * 50)
    print("Test 1.3: ModelWrapper with CNN")
    print("=" * 50)
    
    model = create_simple_cnn(num_classes=10)
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(3, 32, 32), batch_size=16, device=device)
    
    print("Number of ReLU layers: %d" % wrapper.num_relus)
    assert wrapper.num_relus == 2
    
    x = torch.randn(2, 3, 32, 32, device=device)
    direction = normalize_direction(torch.randn(2, 3, 32, 32, device=device))
    
    state = wrapper.get_region_state(x, direction)
    
    print("Logits shape: %s" % str(state.logits.shape))
    print("Lambda to boundary: %s" % state.lambda_to_boundary)
    
    assert state.logits.shape == (2, 10)
    assert state.lambda_to_boundary.shape == (2,)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_forward_simple():
    """Test 1.4: Forward Simple"""
    print("=" * 50)
    print("Test 1.4: Forward Simple")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50], output_dim=10)
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=32, device=device)
    
    x = torch.randn(1, 100, device=device)
    logits = wrapper.forward_simple(x)
    
    print("Input shape: %s" % str(x.shape))
    print("Output shape: %s" % str(logits.shape))
    
    assert logits.shape == (1, 10)
    
    x_batch = torch.randn(5, 100, device=device)
    logits_batch = wrapper.forward_simple(x_batch)
    assert logits_batch.shape == (5, 10)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_region_traverser_single():
    """Test 1.5: LinearRegionTraverser (Single Sample)"""
    print("=" * 50)
    print("Test 1.5: LinearRegionTraverser (Single Sample)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(784,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    x = torch.randn(1, 784, device=device)
    direction = torch.randn(1, 784, device=device)
    
    result = traverser.traverse(start=x, direction=direction, max_distance=5.0, max_regions=20)
    
    print("Number of regions traversed: %d" % result.num_regions)
    print("Total distance: %.4f" % result.total_distance)
    
    for i, region in enumerate(result.regions[:5]):
        print("  Region %d: t=[%.4f, %.4f], length=%.4f" % (
            region.region_id, region.entry_t, region.exit_t, region.length))
    
    if result.num_regions > 5:
        print("  ... and %d more regions" % (result.num_regions - 5))
    
    assert result.num_regions > 0
    assert len(result.regions) == result.num_regions
    
    # Verify region continuity
    for i in range(len(result.regions) - 1):
        curr_exit = result.regions[i].exit_t
        next_entry = result.regions[i + 1].entry_t
        assert abs(curr_exit - next_entry) < 1e-3, "Regions should be continuous"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_region_traverser_batch():
    """Test 1.6: LinearRegionTraverser (Batch)"""
    print("=" * 50)
    print("Test 1.6: LinearRegionTraverser (Batch)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10)
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    batch_size = 8
    x = torch.randn(batch_size, 100, device=device)
    direction = torch.randn(batch_size, 100, device=device)
    
    batch_result = traverser.traverse_batch(
        starts=x, 
        directions=direction, 
        max_distance=3.0, 
        max_regions=15
    )
    
    print("Batch size: %d" % batch_result.batch_size)
    print("Regions per sample: %s" % batch_result.num_regions)
    print("Total distances: %s" % batch_result.total_distances)
    
    assert batch_result.batch_size == batch_size
    assert batch_result.num_regions.shape == (batch_size,)
    assert batch_result.total_distances.shape == (batch_size,)
    
    # Verify each sample's region count is reasonable
    for i in range(batch_size):
        n_regions = batch_result.num_regions[i].item()
        assert n_regions >= 1, "Each sample should have at least 1 region"
        assert n_regions <= 15, "Should not exceed max_regions"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_batch_efficiency():
    """Test 1.7: Batch Processing Efficiency"""
    print("=" * 50)
    print("Test 1.7: Batch Processing Efficiency")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=64, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    batch_size = 16
    x = torch.randn(batch_size, 100, device=device)
    direction = torch.randn(batch_size, 100, device=device)
    
    # Warmup
    _ = traverser.traverse_batch(starts=x[:2], directions=direction[:2], max_distance=1.0, max_regions=5)
    
    # Time batch processing
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    
    batch_result = traverser.traverse_batch(
        starts=x, directions=direction, max_distance=2.0, max_regions=10
    )
    
    if device == "cuda":
        torch.cuda.synchronize()
    batch_time = time.time() - start_time
    
    # Time single processing
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(batch_size):
        _ = traverser.traverse(
            start=x[i:i+1], direction=direction[i:i+1], max_distance=2.0, max_regions=10
        )
    
    if device == "cuda":
        torch.cuda.synchronize()
    single_time = time.time() - start_time
    
    print("Batch processing time: %.4f s" % batch_time)
    print("Single processing time: %.4f s" % single_time)
    print("Speedup: %.2fx" % (single_time / batch_time) if batch_time > 0 else "N/A")
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_deep_network_traversal():
    """Test 1.8: Deep Network Traversal"""
    print("=" * 50)
    print("Test 1.8: Deep Network Traversal")
    print("=" * 50)
    
    # Create a deeper network
    def create_deep_mlp(input_dim=100, hidden_dims=None, output_dim=10):
        if hidden_dims is None:
            hidden_dims = [128, 64, 64, 32, 32, 16]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    model = create_deep_mlp()
    device = get_device()
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    print("Number of ReLU layers: %d" % wrapper.num_relus)
    
    torch.manual_seed(42)
    x = torch.randn(1, 100, device=device)
    direction = torch.randn(1, 100, device=device)
    
    max_distance = 5.0
    result = traverser.traverse(
        start=x, 
        direction=direction, 
        max_distance=max_distance, 
        max_regions=200
    )
    
    print("Number of regions traversed: %d" % result.num_regions)
    print("Total distance: %.4f" % result.total_distance)
    print("Max distance: %.4f" % max_distance)
    print("Distance ratio: %.2f%%" % (result.total_distance / max_distance * 100))
    
    distance_ratio = result.total_distance / max_distance
    success = (
        result.total_distance >= max_distance - 0.01 or
        result.num_regions >= 200 or
        distance_ratio >= 0.80
    )
    
    # Verify region continuity
    gap_count = 0
    for i in range(len(result.regions) - 1):
        curr_exit = result.regions[i].exit_t
        next_entry = result.regions[i + 1].entry_t
        gap = next_entry - curr_exit
        if gap > 0.01:
            gap_count += 1
    
    print("Significant gaps (> 0.01): %d" % gap_count)
    
    assert success, "Deep network traversal should cover most of max_distance"
    assert gap_count == 0, "There should be no significant gaps"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


# =============================================================================
# Phase 2 Tests: DecisionBoundaryDirectionFinder
# =============================================================================

def test_direction_finder_basic():
    """Test 2.1: Basic Direction Finding"""
    print("=" * 50)
    print("Test 2.1: Basic Direction Finding")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = get_device()
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    x = torch.randn(4, 100, device=device)
    directions = finder.find_direction(x)
    
    print("Input shape: %s" % str(x.shape))
    print("Direction shape: %s" % str(directions.shape))
    
    # Check if directions are normalized
    norms = torch.norm(directions.view(4, -1), p=2, dim=1)
    print("Direction norms: %s" % norms)
    
    assert directions.shape == x.shape, "Direction shape should match input shape"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Directions should be normalized"
    
    print("PASSED\n")
    return True


def test_direction_with_info():
    """Test 2.2: Direction with Info"""
    print("=" * 50)
    print("Test 2.2: Direction with Info")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = get_device()
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    x = torch.randn(4, 100, device=device)
    directions, top1_idx, top2_idx, margins = finder.find_direction_with_info(x)
    
    print("Top1 classes: %s" % top1_idx)
    print("Top2 classes: %s" % top2_idx)
    print("Margins (top1 - top2): %s" % margins)
    
    assert directions.shape == x.shape
    assert top1_idx.shape == (4,)
    assert top2_idx.shape == (4,)
    assert margins.shape == (4,)
    
    # Top1 and Top2 should be different
    assert torch.all(top1_idx != top2_idx), "Top1 and Top2 should be different"
    
    # Margin should be non-negative (top1 > top2)
    assert torch.all(margins >= 0), "Margins should be non-negative"
    
    print("PASSED\n")
    return True


def test_direction_moves_toward_boundary():
    """Test 2.3: Direction Moves Toward Boundary"""
    print("=" * 50)
    print("Test 2.3: Direction Moves Toward Boundary")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = get_device()
    model.to(device)
    model.eval()
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    torch.manual_seed(42)
    x = torch.randn(1, 100, device=device)
    direction = finder.find_direction(x)
    
    # Calculate original margin
    with torch.no_grad():
        logits_original = model(x)
        top2_vals, top2_idx = torch.topk(logits_original, k=2, dim=1)
        margin_original = (top2_vals[0, 0] - top2_vals[0, 1]).item()
    
    print("Original margin (top1 - top2): %.4f" % margin_original)
    print("Top1 class: %d, Top2 class: %d" % (top2_idx[0, 0].item(), top2_idx[0, 1].item()))
    
    # Test multiple step sizes - for small steps, margin should decrease
    # (due to gradient direction, at least for very small steps)
    step_sizes = [0.01, 0.05, 0.1, 0.2]
    margin_decreased = False
    
    for step in step_sizes:
        x_moved = x + step * direction
        with torch.no_grad():
            logits_moved = model(x_moved)
            top2_vals_moved, _ = torch.topk(logits_moved, k=2, dim=1)
            margin_moved = (top2_vals_moved[0, 0] - top2_vals_moved[0, 1]).item()
        
        print("  step=%.2f: margin=%.4f (change=%.4f)" % (step, margin_moved, margin_moved - margin_original))
        
        if margin_moved < margin_original:
            margin_decreased = True
            break
    
    assert margin_decreased, "At least one small step should reduce margin"
    
    print("PASSED\n")
    return True


def test_integration_with_traverser():
    """Test 2.4: Integration with Traverser"""
    print("=" * 50)
    print("Test 2.4: Integration with Traverser")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10)
    device = get_device()
    
    # Create direction finder
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    # Create model wrapper and traverser
    wrapper = ModelWrapper(model, input_shape=(100,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    # Test data
    torch.manual_seed(789)
    x = torch.randn(1, 100, device=device)
    
    # Calculate direction toward decision boundary
    direction, top1, top2, margin = finder.find_direction_with_info(x)
    
    print("Initial prediction: class %d (margin to class %d: %.4f)" % (
        top1.item(), top2.item(), margin.item()))
    
    # Traverse linear regions along that direction
    result = traverser.traverse(
        start=x,
        direction=direction,
        max_distance=5.0,
        max_regions=30,
        normalize_dir=False
    )
    
    print("Traversed %d regions" % result.num_regions)
    print("Total distance: %.4f" % result.total_distance)
    
    # Check if final prediction changed
    if result.num_regions > 0:
        last_region = result.regions[-1]
        end_point = x + last_region.exit_t * direction
        
        with torch.no_grad():
            end_logits = model(end_point.to(device))
            end_class = end_logits.argmax(dim=1).item()
            end_top2 = torch.topk(end_logits, k=2, dim=1)
            end_margin = (end_top2.values[0, 0] - end_top2.values[0, 1]).item()
        
        print("End prediction: class %d (margin: %.4f)" % (end_class, end_margin))
        
        if end_class != top1.item():
            print("*** Decision boundary crossed! ***")
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


# =============================================================================
# Phase 3 Tests: RegionPropertyAnalyzer
# =============================================================================

def test_jacobian_computation():
    """Test 3.1: Jacobian Computation"""
    print("=" * 50)
    print("Test 3.1: Jacobian Computation")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=20, hidden_dims=[15], output_dim=5)
    device = get_device()
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    x = torch.randn(2, 20, device=device)
    jacobian = analyzer.compute_jacobian_batch(x)
    
    print("Input: batch=2, dim=20, Output: dim=5")
    print("Jacobian shape: %s" % str(jacobian.shape))
    
    assert jacobian.shape == (2, 5, 20), "Jacobian shape should be (batch, output_dim, input_dim)"
    
    # Validate with finite differences
    eps = 1e-4
    x0 = x[0:1].clone()
    
    numerical_jacobian = torch.zeros(5, 20, device=device)
    
    with torch.no_grad():
        for j in range(20):
            x_plus = x0.clone()
            x_plus[0, j] += eps
            x_minus = x0.clone()
            x_minus[0, j] -= eps
            
            f_plus = model(x_plus)
            f_minus = model(x_minus)
            numerical_jacobian[:, j] = (f_plus[0] - f_minus[0]) / (2 * eps)
    
    analytical_jacobian = jacobian[0]
    
    diff = torch.abs(numerical_jacobian - analytical_jacobian).max().item()
    print("Max difference between numerical and analytical: %.6f" % diff)
    
    assert diff < 1e-3, "Jacobian should match numerical gradient, got diff=%.6f" % diff
    
    print("PASSED\n")
    return True


def test_jacobian_norms():
    """Test 3.2: Jacobian Norms (Batch)"""
    print("=" * 50)
    print("Test 3.2: Jacobian Norms (Batch)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = get_device()
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    batch_size = 4
    x = torch.randn(batch_size, 50, device=device)
    fro_norms, spec_norms = analyzer.compute_jacobian_norms_batch(x)
    
    print("Batch size: %d" % batch_size)
    print("Frobenius norms: %s" % fro_norms)
    print("Spectral norms: %s" % spec_norms)
    
    assert fro_norms.shape == (batch_size,)
    assert spec_norms.shape == (batch_size,)
    
    # Frobenius >= spectral norm
    assert torch.all(fro_norms >= spec_norms - 1e-5)
    assert torch.all(fro_norms > 0)
    
    print("PASSED\n")
    return True


def test_loss_batch():
    """Test 3.3: Loss Computation (Batch)"""
    print("=" * 50)
    print("Test 3.3: Loss Computation (Batch)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, output_dim=10)
    device = get_device()
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    batch_size = 4
    x = torch.randn(batch_size, 50, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    losses = analyzer.compute_loss_batch(x, labels)
    
    print("Losses: %s" % losses)
    
    assert losses.shape == (batch_size,)
    assert torch.all(losses >= 0)
    
    # Verify against manual calculation
    with torch.no_grad():
        logits = model(x)
        expected = nn.functional.cross_entropy(logits, labels, reduction='none')
    
    assert torch.allclose(losses, expected)
    
    print("PASSED\n")
    return True


def test_full_traversal_analysis():
    """Test 3.4: Full Traversal Analysis"""
    print("=" * 50)
    print("Test 3.4: Full Traversal Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = get_device()
    
    wrapper = ModelWrapper(model, input_shape=(50,), batch_size=16, device=device)
    traverser = LinearRegionTraverser(wrapper)
    analyzer = RegionPropertyAnalyzer(model, device, num_samples_per_region=3)
    
    torch.manual_seed(456)
    x = torch.randn(1, 50, device=device)
    direction = normalize_direction(torch.randn(1, 50, device=device))
    label = 7
    
    traversal = traverser.traverse(x, direction, max_distance=3.0, max_regions=10)
    
    print("Traversed %d regions" % traversal.num_regions)
    
    props = analyzer.analyze_traversal(traversal, x, direction, label)
    
    print("Analysis results:")
    print("  Num regions: %d" % props.num_regions)
    print("  Mean Jacobian norm: %.4f" % props.mean_jacobian_norm)
    print("  Max Jacobian norm: %.4f" % props.max_jacobian_norm)
    print("  Mean Jacobian diff: %.4f" % props.mean_jacobian_diff)
    print("  Mean loss diff: %.4f" % props.mean_loss_diff)
    print("  Total loss change: %.4f" % props.total_loss_change)
    
    assert props.num_regions == traversal.num_regions
    assert len(props.region_properties) == traversal.num_regions
    assert len(props.adjacent_diffs) == max(0, traversal.num_regions - 1)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


# =============================================================================
# Phase 4 Tests: LinearRegionAnalyzer
# =============================================================================

def test_single_analysis():
    """Test 4.1: Single Sample Analysis"""
    print("=" * 50)
    print("Test 4.1: Single Sample Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = get_device()
    
    analyzer = LinearRegionAnalyzer(
        model=model,
        input_shape=(50,),
        device=device,
        batch_size=16
    )
    
    torch.manual_seed(42)
    x = torch.randn(1, 50, device=device)
    label = 3
    
    result = analyzer.analyze(x, label, max_distance=0.05, max_regions=20)
    
    print("Predicted class: %d" % result.predicted_class)
    print("Second class: %d" % result.second_class)
    print("Margin: %.4f" % result.margin)
    print("Num regions: %d" % result.num_regions)
    print("Total distance: %.4f" % result.total_distance)
    print("Mean Jacobian norm: %.4f" % result.mean_jacobian_norm)
    print("Mean Jacobian diff: %.4f" % result.mean_jacobian_diff)
    print("Total loss change: %.4f" % result.total_loss_change)
    print("Crossed boundary: %s" % result.crossed_decision_boundary)
    print("Final class: %d" % result.final_class)
    
    assert result.num_regions > 0
    assert result.mean_jacobian_norm > 0
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_batch_analysis():
    """Test 4.2: Batch Analysis"""
    print("=" * 50)
    print("Test 4.2: Batch Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = get_device()
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(50,), device=device)
    
    batch_size = 4
    torch.manual_seed(456)
    x = torch.randn(batch_size, 50, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    batch_result = analyzer.analyze_batch(x, labels, max_distance=3.0, max_regions=15)
    
    print("Batch size: %d" % batch_result.batch_size)
    print("Mean num regions: %.2f" % batch_result.mean_num_regions)
    print("Mean Jacobian norm: %.4f" % batch_result.mean_jacobian_norm)
    print("Mean Jacobian diff: %.4f" % batch_result.mean_jacobian_diff)
    print("Boundary crossing rate: %.2f" % batch_result.boundary_crossing_rate)
    
    print("\nPer-sample results:")
    for i, r in enumerate(batch_result.results):
        print("  Sample %d: pred=%d, regions=%d, jac=%.4f, crossed=%s" % (
            i, r.predicted_class, r.num_regions, r.mean_jacobian_norm, r.crossed_decision_boundary))
    
    assert len(batch_result.results) == batch_size
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_compare_directions():
    """Test 4.3: Compare Directions"""
    print("=" * 50)
    print("Test 4.3: Compare Directions")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = get_device()
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(50,), device=device)
    
    torch.manual_seed(111)
    x = torch.randn(1, 50, device=device)
    
    comparison = analyzer.compare_directions(
        x, label=4, 
        max_distance=5.0, 
        max_regions=20,
        num_random=3
    )
    
    print("Boundary direction:")
    print("  Num regions: %d" % comparison['boundary_direction']['num_regions'])
    print("  Mean Jacobian norm: %.4f" % comparison['boundary_direction']['mean_jacobian_norm'])
    print("  Mean Jacobian diff: %.4f" % comparison['boundary_direction']['mean_jacobian_diff'])
    print("  Crossed: %s" % comparison['boundary_direction']['crossed_boundary'])
    
    print("\nRandom directions (avg of 3):")
    print("  Num regions: %.2f ± %.2f" % (
        comparison['random_directions']['num_regions_mean'],
        comparison['random_directions']['num_regions_std']))
    print("  Mean Jacobian norm: %.4f" % comparison['random_directions']['mean_jacobian_norm'])
    print("  Mean Jacobian diff: %.4f" % comparison['random_directions']['mean_jacobian_diff'])
    print("  Crossing rate: %.2f" % comparison['random_directions']['crossing_rate'])
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_cnn_model():
    """Test 4.4: CNN Model"""
    print("=" * 50)
    print("Test 4.4: CNN Model")
    print("=" * 50)
    
    # Simple CNN
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 10)
    )
    
    device = get_device()
    
    analyzer = LinearRegionAnalyzer(
        model=model,
        input_shape=(1, 8, 8),
        device=device
    )
    
    torch.manual_seed(333)
    x = torch.randn(1, 1, 8, 8, device=device)
    
    result = analyzer.analyze(x, label=5, max_distance=2.0, max_regions=10)
    
    print("CNN analysis:")
    print("  Num regions: %d" % result.num_regions)
    print("  Mean Jacobian norm: %.4f" % result.mean_jacobian_norm)
    print("  Predicted: %d, Final: %d" % (result.predicted_class, result.final_class))
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Running All Tests for Linear Region Analysis Toolkit")
    print("=" * 70 + "\n")
    
    tests = [
        # Phase 1: ModelWrapper and LinearRegionTraverser
        ("1.1", test_direction_normalization),
        ("1.2", test_model_wrapper_mlp),
        ("1.3", test_model_wrapper_cnn),
        ("1.4", test_forward_simple),
        ("1.5", test_region_traverser_single),
        ("1.6", test_region_traverser_batch),
        ("1.7", test_batch_efficiency),
        ("1.8", test_deep_network_traversal),
        
        # Phase 2: DecisionBoundaryDirectionFinder
        ("2.1", test_direction_finder_basic),
        ("2.2", test_direction_with_info),
        ("2.3", test_direction_moves_toward_boundary),
        ("2.4", test_integration_with_traverser),
        
        # Phase 3: RegionPropertyAnalyzer
        ("3.1", test_jacobian_computation),
        ("3.2", test_jacobian_norms),
        ("3.3", test_loss_batch),
        ("3.4", test_full_traversal_analysis),
        
        # Phase 4: LinearRegionAnalyzer
        ("4.1", test_single_analysis),
        ("4.2", test_batch_analysis),
        ("4.3", test_compare_directions),
        ("4.4", test_cnn_model),
    ]
    
    passed = 0
    failed = 0
    
    for test_id, test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print("FAILED with error: %s" % e)
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print("Results: %d passed, %d failed" % (passed, failed))
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
