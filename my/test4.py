# -*- coding: utf-8 -*-
"""
Phase 4 测试脚本：整合分析器
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
    normalize_direction,
    LinearRegionAnalyzer
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


def test_single_analysis():
    """测试单样本分析"""
    print("=" * 50)
    print("Test 1: Single Sample Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    print("Predicted class: %d" % result. predicted_class)
    print("Second class: %d" % result.second_class)
    print("Margin: %.4f" % result. margin)
    print("Num regions: %d" % result.num_regions)
    print("Total distance: %.4f" % result.total_distance)
    print("Mean Jacobian norm: %.4f" % result.mean_jacobian_norm)
    print("Mean Jacobian diff: %.4f" % result.mean_jacobian_diff)
    print("Total loss change: %.4f" % result. total_loss_change)
    print("Crossed boundary: %s" % result.crossed_decision_boundary)
    print("Final class: %d" % result.final_class)
    
    assert result.num_regions > 0
    assert result. mean_jacobian_norm > 0
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_single_analysis_with_details():
    """测试保留详细结果"""
    print("=" * 50)
    print("Test 2: Analysis with Details")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(50,), device=device)
    
    torch.manual_seed(123)
    x = torch.randn(1, 50, device=device)
    
    result = analyzer. analyze(x, label=5, max_distance=3.0, keep_details=True)
    
    print("Num regions: %d" % result.num_regions)
    
    # 检查详细结果
    assert result.traversal is not None, "Traversal should be kept"
    assert result.properties is not None, "Properties should be kept"
    
    print("Traversal regions: %d" % len(result.traversal.regions))
    print("Property regions: %d" % len(result. properties.region_properties))
    
    # 打印每个区域的性质
    print("\nPer-region details:")
    for rp in result.properties.region_properties[:5]:
        print("  Region %d: t=[%.4f, %.4f], Jac=%.4f, Loss=%.4f" % (
            rp. region_id, rp.entry_t, rp.exit_t, 
            rp. jacobian_frobenius_norm, rp. mean_loss))
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_batch_analysis():
    """测试批量分析"""
    print("=" * 50)
    print("Test 3: Batch Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(50,), device=device)
    
    batch_size = 4
    torch.manual_seed(456)
    x = torch.randn(batch_size, 50, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    batch_result = analyzer. analyze_batch(x, labels, max_distance=3.0, max_regions=15)
    
    print("Batch size: %d" % batch_result.batch_size)
    print("Mean num regions: %.2f" % batch_result.mean_num_regions)
    print("Mean Jacobian norm: %.4f" % batch_result.mean_jacobian_norm)
    print("Mean Jacobian diff: %.4f" % batch_result.mean_jacobian_diff)
    print("Boundary crossing rate: %.2f" % batch_result. boundary_crossing_rate)
    
    print("\nPer-sample results:")
    for i, r in enumerate(batch_result.results):
        print("  Sample %d: pred=%d, regions=%d, jac=%.4f, crossed=%s" % (
            i, r.predicted_class, r.num_regions, r.mean_jacobian_norm, r.crossed_decision_boundary))
    
    assert len(batch_result. results) == batch_size
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_random_direction():
    """测试随机方向分析"""
    print("=" * 50)
    print("Test 4: Random Direction Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(50,), device=device)
    
    torch.manual_seed(789)
    x = torch. randn(1, 50, device=device)
    
    result = analyzer.analyze_with_random_direction(x, label=2, max_distance=3.0, seed=42)
    
    print("Random direction result:")
    print("  Num regions: %d" % result.num_regions)
    print("  Mean Jacobian norm: %.4f" % result.mean_jacobian_norm)
    print("  Crossed boundary: %s" % result.crossed_decision_boundary)
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_compare_directions():
    """测试方向比较"""
    print("=" * 50)
    print("Test 5: Compare Directions")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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


def test_no_label():
    """测试不提供 label 的情况"""
    print("=" * 50)
    print("Test 6: Analysis without Label")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(50,), device=device)
    
    torch.manual_seed(222)
    x = torch. randn(1, 50, device=device)
    
    # 不提供 label
    result = analyzer.analyze(x, label=None, max_distance=3.0)
    
    print("Analysis without label:")
    print("  Num regions: %d" % result.num_regions)
    print("  Mean Jacobian norm: %.4f" % result. mean_jacobian_norm)
    print("  Mean loss diff: %.4f (should be 0)" % result.mean_loss_diff)
    
    # 没有 label 时，loss 相关统计应该为 0
    assert result.mean_loss_diff == 0.0
    assert result.total_loss_change == 0.0
    
    analyzer. cleanup()
    print("PASSED\n")
    return True


def test_cnn_model():
    """测试 CNN 模型"""
    print("=" * 50)
    print("Test 7: CNN Model")
    print("=" * 50)
    
    # 简单 CNN
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn. AdaptiveAvgPool2d(4),
        nn. Flatten(),
        nn. Linear(8 * 4 * 4, 10)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analyzer = LinearRegionAnalyzer(
        model=model,
        input_shape=(1, 8, 8),  # 小图像
        device=device
    )
    
    torch.manual_seed(333)
    x = torch. randn(1, 1, 8, 8, device=device)
    
    result = analyzer.analyze(x, label=5, max_distance=2.0, max_regions=10)
    
    print("CNN analysis:")
    print("  Num regions: %d" % result.num_regions)
    print("  Mean Jacobian norm: %.4f" % result.mean_jacobian_norm)
    print("  Predicted: %d, Final: %d" % (result.predicted_class, result.final_class))
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def test_efficiency():
    """测试效率"""
    print("=" * 50)
    print("Test 8: Efficiency")
    print("=" * 50)
    
    import time
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[64, 32], output_dim=10)
    device = "cuda" if torch. cuda.is_available() else "cpu"
    
    analyzer = LinearRegionAnalyzer(model, input_shape=(100,), device=device)
    
    # 预热
    x_warm = torch.randn(1, 100, device=device)
    _ = analyzer.analyze(x_warm, label=0, max_distance=1.0, max_regions=5)
    
    # 单样本计时
    x = torch.randn(1, 100, device=device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    
    result = analyzer.analyze(x, label=3, max_distance=5.0, max_regions=30)
    
    if device == "cuda":
        torch.cuda.synchronize()
    single_time = time. time() - start
    
    print("Single sample: %.4f s, %d regions" % (single_time, result.num_regions))
    
    # 批量计时
    batch_size = 8
    x_batch = torch. randn(batch_size, 100, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    
    batch_result = analyzer. analyze_batch(x_batch, labels, max_distance=3.0, max_regions=15)
    
    if device == "cuda":
        torch. cuda.synchronize()
    batch_time = time.time() - start
    
    print("Batch (%d samples): %.4f s, %.4f s/sample" % (
        batch_size, batch_time, batch_time / batch_size))
    print("Mean regions: %.2f" % batch_result. mean_num_regions)
    
    analyzer.cleanup()
    print("PASSED\n")
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("Running Phase 4 Tests: Integrated Analyzer")
    print("=" * 60 + "\n")
    
    tests = [
        test_single_analysis,
        test_single_analysis_with_details,
        test_batch_analysis,
        test_random_direction,
        test_compare_directions,
        test_no_label,
        test_cnn_model,
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