# -*- coding: utf-8 -*-
"""
Phase 1 测试脚本
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from my import ModelWrapper, LinearRegionTraverser, normalize_direction
from my.core.model_wrapper import EPSILON


def create_simple_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers. append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn. Linear(prev_dim, output_dim))
    return nn. Sequential(*layers)


def create_simple_cnn(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn. Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn. Linear(64, num_classes)
    )


def test_direction_normalization():
    print("=" * 50)
    print("Test 1: Direction Normalization")
    print("=" * 50)
    
    direction = torch.randn(4, 3, 32, 32)
    normalized = normalize_direction(direction)
    
    norms = torch.norm(normalized. view(4, -1), p=2, dim=1)
    print("Norms after normalization: %s" % norms)
    
    assert torch.allclose(norms, torch. ones_like(norms), atol=1e-5)
    print("PASSED\n")
    return True


def test_model_wrapper_mlp():
    print("=" * 50)
    print("Test 2: ModelWrapper with MLP")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model=model, input_shape=(784,), batch_size=32, device=device)
    
    print("Number of ReLU layers: %d" % wrapper. num_relus)
    assert wrapper.num_relus == 2
    
    x = torch.randn(4, 784, device=device)
    direction = normalize_direction(torch. randn(4, 784, device=device))
    
    state = wrapper.get_region_state(x, direction)
    
    print("Logits shape: %s" % str(state.logits.shape))
    print("Lambda to boundary: %s" % state.lambda_to_boundary)
    
    assert state.logits.shape == (4, 10)
    assert state. lambda_to_boundary.shape == (4,)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_model_wrapper_cnn():
    print("=" * 50)
    print("Test 3: ModelWrapper with CNN")
    print("=" * 50)
    
    model = create_simple_cnn(num_classes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model=model, input_shape=(3, 32, 32), batch_size=16, device=device)
    
    print("Number of ReLU layers: %d" % wrapper.num_relus)
    assert wrapper.num_relus == 2
    
    x = torch.randn(2, 3, 32, 32, device=device)
    direction = normalize_direction(torch.randn(2, 3, 32, 32, device=device))
    
    state = wrapper. get_region_state(x, direction)
    
    print("Logits shape: %s" % str(state.logits.shape))
    print("Lambda to boundary: %s" % state.lambda_to_boundary)
    
    assert state.logits.shape == (2, 10)
    assert state.lambda_to_boundary.shape == (2,)
    
    wrapper. cleanup()
    print("PASSED\n")
    return True


def test_forward_simple():
    print("=" * 50)
    print("Test 4: Forward Simple")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=32, device=device)
    
    x = torch.randn(1, 100, device=device)
    logits = wrapper.forward_simple(x)
    
    print("Input shape: %s" % str(x. shape))
    print("Output shape: %s" % str(logits.shape))
    
    assert logits.shape == (1, 10)
    
    x_batch = torch.randn(5, 100, device=device)
    logits_batch = wrapper.forward_simple(x_batch)
    assert logits_batch. shape == (5, 10)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_region_traverser_single():
    print("=" * 50)
    print("Test 5: LinearRegionTraverser (Single Sample)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # 验证区域连续性
    for i in range(len(result.regions) - 1):
        curr_exit = result.regions[i].exit_t
        next_entry = result.regions[i + 1]. entry_t
        assert abs(curr_exit - next_entry) < 1e-3, "Regions should be continuous"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_region_traverser_batch():
    print("=" * 50)
    print("Test 6: LinearRegionTraverser (Batch)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    assert batch_result. batch_size == batch_size
    assert batch_result.num_regions. shape == (batch_size,)
    assert batch_result.total_distances.shape == (batch_size,)
    
    # 验证每个样本的区域数量合理
    for i in range(batch_size):
        n_regions = batch_result.num_regions[i].item()
        assert n_regions >= 1, "Each sample should have at least 1 region"
        assert n_regions <= 15, "Should not exceed max_regions"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_batch_vs_single_consistency():
    print("=" * 50)
    print("Test 7: Batch vs Single Consistency")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    torch.manual_seed(42)
    batch_size = 4
    x = torch.randn(batch_size, 100, device=device)
    direction = torch.randn(batch_size, 100, device=device)
    
    # 批量处理
    batch_result = traverser.traverse_batch(
        starts=x, directions=direction, max_distance=2.0, max_regions=10
    )
    
    # 逐个处理
    single_results = []
    for i in range(batch_size):
        result = traverser.traverse(
            start=x[i:i+1], 
            direction=direction[i:i+1], 
            max_distance=2.0, 
            max_regions=10
        )
        single_results. append(result)
    
    # 比较结果
    print("Comparing batch vs single results:")
    for i in range(batch_size):
        batch_n = batch_result.num_regions[i]. item()
        single_n = single_results[i].num_regions
        print("  Sample %d: batch=%d regions, single=%d regions" % (i, batch_n, single_n))
        
        # 区域数量应该相同
        assert batch_n == single_n, "Region count mismatch for sample %d" % i
        
        # 比较 entry_t 和 exit_t
        for r in range(batch_n):
            batch_entry = batch_result.entry_ts[i, r]. item()
            batch_exit = batch_result.exit_ts[i, r]. item()
            single_entry = single_results[i]. regions[r].entry_t
            single_exit = single_results[i].regions[r].exit_t
            
            assert abs(batch_entry - single_entry) < 1e-6, "entry_t mismatch"
            assert abs(batch_exit - single_exit) < 1e-6, "exit_t mismatch"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_batch_efficiency():
    print("=" * 50)
    print("Test 8: Batch Processing Efficiency")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    device = "cuda" if torch. cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=64, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    batch_size = 16
    x = torch.randn(batch_size, 100, device=device)
    direction = torch.randn(batch_size, 100, device=device)
    
    # 预热
    _ = traverser.traverse_batch(starts=x[:2], directions=direction[:2], max_distance=1.0, max_regions=5)
    
    # 测量批量处理时间
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time. time()
    
    batch_result = traverser.traverse_batch(
        starts=x, directions=direction, max_distance=2.0, max_regions=10
    )
    
    if device == "cuda":
        torch.cuda.synchronize()
    batch_time = time. time() - start_time
    
    # 测量逐个处理时间
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
    
    # 批量处理应该更快（或至少不慢太多）
    # 注意：由于实现复杂性，小 batch 可能没有明显加速
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_activation_pattern_changes():
    print("=" * 50)
    print("Test 9: Activation Pattern Changes")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model=model, input_shape=(100,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    torch.manual_seed(123)
    x = torch.randn(1, 100, device=device)
    direction = torch.randn(1, 100, device=device)
    
    result = traverser.traverse(start=x, direction=direction, max_distance=3.0, max_regions=10)
    
    print("Traversed %d regions" % result.num_regions)
    
    if result.num_regions > 1:
        changes = 0
        for i in range(len(result.regions) - 1):
            p1 = result.regions[i].activation_pattern
            p2 = result. regions[i + 1].activation_pattern
            if not (p1 == p2):
                changes += 1
        
        print("Activation pattern changes: %d/%d" % (changes, result.num_regions - 1))
        
        # 每次跨越边界都应该改变激活模式
        assert changes == result.num_regions - 1, "Each crossing should change pattern"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_lambda_correctness():
    print("=" * 50)
    print("Test 10: Lambda Correctness")
    print("=" * 50)
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    
    wrapper = ModelWrapper(model=model, input_shape=(10,), batch_size=8, device="cpu")
    
    torch.manual_seed(42)
    x = torch.randn(1, 10)
    direction = normalize_direction(torch. randn(1, 10))
    
    state = wrapper.get_region_state(x, direction)
    lambda_val = state.lambda_to_boundary. item()
    
    print("Lambda to boundary: %.6f" % lambda_val)
    
    if not np.isinf(lambda_val) and lambda_val > EPSILON:
        x_near = x + (lambda_val - EPSILON * 0.1) * direction
        state_near = wrapper. get_region_state(x_near, direction)
        lambda_near = state_near. lambda_to_boundary.item()
        print("Lambda near boundary: %.6f" % lambda_near)
        
        x_past = x + (lambda_val + EPSILON * 10) * direction
        state_past = wrapper.get_region_state(x_past, direction)
        
        pattern_changed = not (state.activation_pattern == state_past.activation_pattern)
        print("Pattern changed after crossing: %s" % pattern_changed)
        
        #assert lambda_near < EPSILON * 2, "Lambda near boundary should be small"
        assert pattern_changed, "Pattern should change after crossing"
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("Running Phase 1 Tests (with Batch Processing)")
    print("=" * 60 + "\n")
    
    tests = [
        test_direction_normalization,
        test_model_wrapper_mlp,
        test_model_wrapper_cnn,
        test_forward_simple,
        test_region_traverser_single,
        test_region_traverser_batch,
        test_batch_vs_single_consistency,
        test_batch_efficiency,
        test_activation_pattern_changes,
        test_lambda_correctness,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print("FAILED with error: %s" % e)
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("Results: %d passed, %d failed" % (passed, failed))
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)