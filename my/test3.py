# -*- coding: utf-8 -*-
"""
Phase 3 测试脚本：区域性质计算
"""

import torch
import torch.nn as nn
import sys
import os


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from my.core import (
    ModelWrapper,
    LinearRegionTraverser,
    DecisionBoundaryDirectionFinder,
    RegionPropertyAnalyzer,
    normalize_direction
)


def create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn. Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers. append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def test_jacobian_computation():
    """测试雅可比矩阵计算"""
    print("=" * 50)
    print("Test 1: Jacobian Computation")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=20, hidden_dims=[15], output_dim=5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    x = torch.randn(2, 20, device=device)  # batch=2
    jacobian = analyzer.compute_jacobian_batch(x)
    
    print("Input: batch=2, dim=20, Output: dim=5")
    print("Jacobian shape: %s" % str(jacobian.shape))
    
    assert jacobian.shape == (2, 5, 20), "Jacobian shape should be (batch, output_dim, input_dim)"
    
    # 用有限差分验证（使用更小的 eps 和更宽松的阈值）
    eps = 1e-4
    x0 = x[0:1]. clone()
    
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
    
    diff = torch.abs(numerical_jacobian - analytical_jacobian). max().item()
    print("Max difference between numerical and analytical: %.6f" % diff)
    
    # 放宽阈值到 1e-3
    assert diff < 1e-3, "Jacobian should match numerical gradient, got diff=%.6f" % diff
    
    print("PASSED\n")
    return True


def test_jacobian_norms():
    """测试雅可比范数批量计算"""
    print("=" * 50)
    print("Test 2: Jacobian Norms (Batch)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    batch_size = 4
    x = torch.randn(batch_size, 50, device=device)
    fro_norms, spec_norms = analyzer.compute_jacobian_norms_batch(x)
    
    print("Batch size: %d" % batch_size)
    print("Frobenius norms: %s" % fro_norms)
    print("Spectral norms: %s" % spec_norms)
    
    assert fro_norms.shape == (batch_size,)
    assert spec_norms.shape == (batch_size,)
    
    # Frobenius >= 谱范数
    assert torch.all(fro_norms >= spec_norms - 1e-5)
    assert torch.all(fro_norms > 0)
    
    print("PASSED\n")
    return True


def test_loss_batch():
    """测试批量 loss 计算"""
    print("=" * 50)
    print("Test 3: Loss Computation (Batch)")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, output_dim=10)
    device = "cuda" if torch. cuda.is_available() else "cpu"
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    batch_size = 4
    x = torch.randn(batch_size, 50, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    losses = analyzer.compute_loss_batch(x, labels)
    
    print("Losses: %s" % losses)
    
    assert losses.shape == (batch_size,)
    assert torch.all(losses >= 0)
    
    # 验证与手动计算一致
    with torch.no_grad():
        logits = model(x)
        expected = nn.functional.cross_entropy(logits, labels, reduction='none')
    
    assert torch. allclose(losses, expected)
    
    print("PASSED\n")
    return True


def test_full_traversal_analysis():
    """测试完整遍历分析"""
    print("=" * 50)
    print("Test 4: Full Traversal Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    print("\nPer-region properties:")
    for rp in props.region_properties[:5]:
        print("  Region %d: Fro=%.4f, Spec=%.4f, Loss=%.4f" % (
            rp.region_id, rp.jacobian_frobenius_norm, rp. jacobian_spectral_norm, rp.mean_loss))
    
    assert props.num_regions == traversal.num_regions
    assert len(props.region_properties) == traversal.num_regions
    assert len(props.adjacent_diffs) == max(0, traversal.num_regions - 1)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_integration_with_direction_finder():
    """测试与方向查找器的集成"""
    print("=" * 50)
    print("Test 5: Integration with Direction Finder")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    wrapper = ModelWrapper(model, input_shape=(50,), batch_size=16, device=device)
    traverser = LinearRegionTraverser(wrapper)
    analyzer = RegionPropertyAnalyzer(model, device)
    
    torch.manual_seed(789)
    x = torch.randn(1, 50, device=device)
    
    with torch.no_grad():
        logits = model(x)
        label = logits.argmax(dim=1).item()
    
    print("Predicted label: %d" % label)
    
    # 1. 找方向
    direction, top1, top2, margin = finder.find_direction_with_info(x)
    print("Direction: top1=%d, top2=%d, margin=%.4f" % (top1. item(), top2. item(), margin.item()))
    
    # 2. 遍历
    traversal = traverser.traverse(x, direction, max_distance=5.0, max_regions=20, normalize_dir=False)
    print("Traversed %d regions, distance=%.4f" % (traversal. num_regions, traversal.total_distance))
    
    # 3. 分析
    props = analyzer.analyze_traversal(traversal, x, direction, label)
    
    print("Results:")
    print("  Mean Jacobian norm: %.4f" % props.mean_jacobian_norm)
    print("  Mean Jacobian diff: %.4f" % props.mean_jacobian_diff)
    print("  Total loss change: %.4f" % props. total_loss_change)
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_batch_analysis():
    """测试批量分析"""
    print("=" * 50)
    print("Test 6: Batch Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wrapper = ModelWrapper(model, input_shape=(50,), batch_size=16, device=device)
    traverser = LinearRegionTraverser(wrapper)
    analyzer = RegionPropertyAnalyzer(model, device, num_samples_per_region=2)
    
    batch_size = 4
    torch.manual_seed(111)
    x = torch.randn(batch_size, 50, device=device)
    direction = normalize_direction(torch.randn(batch_size, 50, device=device))
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    batch_traversal = traverser.traverse_batch(x, direction, max_distance=2.0, max_regions=8)
    
    print("Batch size: %d" % batch_traversal.batch_size)
    print("Regions per sample: %s" % batch_traversal.num_regions. tolist())
    
    all_props = analyzer.analyze_batch(batch_traversal, x, direction, labels)
    
    print("\nPer-sample results:")
    for i, props in enumerate(all_props):
        print("  Sample %d: %d regions, mean_jac=%.4f, loss_diff=%.4f" % (
            i, props. num_regions, props.mean_jacobian_norm, props. mean_loss_diff))
    
    assert len(all_props) == batch_size
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_efficiency():
    """测试计算效率"""
    print("=" * 50)
    print("Test 7: Efficiency Test")
    print("=" * 50)
    
    import time
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    device = "cuda" if torch. cuda.is_available() else "cpu"
    
    analyzer = RegionPropertyAnalyzer(model, device)
    
    # 预热
    x_warm = torch.randn(2, 100, device=device)
    _ = analyzer.compute_jacobian_norms_batch(x_warm)
    
    # 测试批量计算
    batch_size = 16
    x = torch.randn(batch_size, 100, device=device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    
    fro_norms, spec_norms = analyzer.compute_jacobian_norms_batch(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print("Batch size: %d" % batch_size)
    print("Time for Jacobian norms: %.4f s" % elapsed)
    print("Time per sample: %.4f s" % (elapsed / batch_size))
    
    print("PASSED\n")
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("Running Phase 3 Tests: Region Property Analysis")
    print("=" * 60 + "\n")
    
    tests = [
        test_jacobian_computation,
        test_jacobian_norms,
        test_loss_batch,
        test_full_traversal_analysis,
        test_integration_with_direction_finder,
        test_batch_analysis,
        test_efficiency,
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