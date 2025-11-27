# -*- coding: utf-8 -*-
"""
Tests for SimpleLinearRegionAnalyzer

Validates the 4 core metrics of the simplified analyzer.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my import SimpleLinearRegionAnalyzer, SimpleAnalysisResult


def create_simple_mlp(input_dim=100, hidden_dims=None, output_dim=10):
    """Create a simple MLP test model"""
    if hidden_dims is None:
        hidden_dims = [64, 32]
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def create_simple_cnn(num_classes=10):
    """Create a simple CNN test model"""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, num_classes)
    )


def get_device():
    """Get available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_simple_analyzer_creation():
    """Test simplified analyzer creation"""
    print("=" * 50)
    print("Test: SimpleLinearRegionAnalyzer Creation")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30], output_dim=10)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(50,),
        device=device
    )
    
    print(f"Analyzer created successfully")
    print(f"Device: {analyzer.device}")
    print(f"Input shape: {analyzer.input_shape}")
    print(f"Number of ReLU layers: {analyzer._num_relus}")
    
    assert analyzer._num_relus == 1
    print("PASSED\n")
    return True


def test_gradient_norm_computation():
    """Test gradient norm computation"""
    print("=" * 50)
    print("Test: Gradient Norm Computation")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=20, hidden_dims=[10], output_dim=5)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(20,),
        device=device
    )
    
    torch.manual_seed(42)
    x = torch.randn(1, 20, device=device)
    
    grad_norm = analyzer._compute_gradient_norm(x)
    
    print(f"Gradient norm: {grad_norm:.4f}")
    
    assert grad_norm > 0, "Gradient norm should be positive"
    assert not torch.isnan(torch.tensor(grad_norm)), "Gradient norm should not be NaN"
    
    print("PASSED\n")
    return True


def test_activation_pattern():
    """Test activation pattern retrieval"""
    print("=" * 50)
    print("Test: Activation Pattern")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=20, hidden_dims=[10, 5], output_dim=3)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(20,),
        device=device
    )
    
    torch.manual_seed(42)
    x1 = torch.randn(1, 20, device=device)
    x2 = torch.randn(1, 20, device=device)
    
    pattern1 = analyzer._get_activation_pattern(x1)
    pattern2 = analyzer._get_activation_pattern(x2)
    
    print(f"Number of ReLU layers: {len(pattern1)}")
    print(f"Pattern 1 same as Pattern 2: {analyzer._pattern_same(pattern1, pattern2)}")
    print(f"Pattern 1 same as itself: {analyzer._pattern_same(pattern1, pattern1)}")
    
    assert len(pattern1) == 2, "Should have 2 ReLU layers"
    assert analyzer._pattern_same(pattern1, pattern1), "Same pattern should be equal"
    
    print("PASSED\n")
    return True


def test_boundary_detection():
    """Test boundary detection"""
    print("=" * 50)
    print("Test: Boundary Detection")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=20, hidden_dims=[15], output_dim=5)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(20,),
        device=device
    )
    
    torch.manual_seed(42)
    x = torch.randn(1, 20, device=device)
    direction = torch.randn(1, 20, device=device)
    
    lambda_val = analyzer._compute_lambda_to_boundary(x, direction)
    
    print(f"Lambda to boundary: {lambda_val:.4f}")
    
    # Lambda 应该是正数或无穷大
    assert lambda_val > 0, "Lambda should be positive"
    
    print("PASSED\n")
    return True


def test_single_sample_analysis():
    """Test single sample analysis"""
    print("=" * 50)
    print("Test: Single Sample Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=50, hidden_dims=[30, 20], output_dim=10)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(50,),
        device=device
    )
    
    torch.manual_seed(42)
    x = torch.randn(1, 50, device=device)
    
    # 使用 analyze 方法（自动计算方向）
    result = analyzer.analyze(x, label=3, max_distance=1.0, max_regions=20)
    
    print(f"Result type: {type(result)}")
    print(f"Number of regions: {result.num_regions}")
    print(f"Mean gradient norm: {result.mean_gradient_norm:.4f}")
    print(f"Mean gradient norm change: {result.mean_gradient_norm_change:.4f}")
    print(f"Mean loss change: {result.mean_loss_change:.4f}")
    
    assert isinstance(result, SimpleAnalysisResult)
    assert result.num_regions >= 1
    assert result.mean_gradient_norm >= 0
    
    print("PASSED\n")
    return True


def test_direction_analysis():
    """Test analysis with specified direction"""
    print("=" * 50)
    print("Test: Direction Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=30, hidden_dims=[20], output_dim=5)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(30,),
        device=device
    )
    
    torch.manual_seed(123)
    x = torch.randn(1, 30, device=device)
    direction = torch.randn(1, 30, device=device)
    
    result = analyzer.analyze_direction(
        x, direction, label=2, 
        max_distance=2.0, max_regions=30
    )
    
    print(f"Number of regions: {result.num_regions}")
    print(f"Mean gradient norm: {result.mean_gradient_norm:.4f}")
    print(f"Mean gradient norm change: {result.mean_gradient_norm_change:.4f}")
    print(f"Mean loss change: {result.mean_loss_change:.4f}")
    
    assert result.num_regions >= 1
    
    print("PASSED\n")
    return True


def test_batch_analysis():
    """Test batch analysis"""
    print("=" * 50)
    print("Test: Batch Analysis")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=30, hidden_dims=[20], output_dim=5)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(30,),
        device=device
    )
    
    batch_size = 4
    torch.manual_seed(456)
    x_batch = torch.randn(batch_size, 30, device=device)
    directions = torch.randn(batch_size, 30, device=device)
    labels = torch.randint(0, 5, (batch_size,), device=device)
    
    results = analyzer.analyze_batch(
        x_batch, directions, labels,
        max_distance=1.0, max_regions=15
    )
    
    print(f"Number of results: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"  Sample {i}: regions={result.num_regions}, "
              f"grad_norm={result.mean_gradient_norm:.4f}, "
              f"grad_change={result.mean_gradient_norm_change:.4f}")
    
    assert len(results) == batch_size
    for result in results:
        assert isinstance(result, SimpleAnalysisResult)
        assert result.num_regions >= 1
    
    print("PASSED\n")
    return True


def test_cnn_model():
    """Test CNN model"""
    print("=" * 50)
    print("Test: CNN Model")
    print("=" * 50)
    
    model = create_simple_cnn(num_classes=10)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(3, 8, 8),
        device=device
    )
    
    print(f"Number of ReLU layers: {analyzer._num_relus}")
    
    torch.manual_seed(789)
    x = torch.randn(1, 3, 8, 8, device=device)
    
    result = analyzer.analyze(x, label=5, max_distance=1.0, max_regions=10)
    
    print(f"Number of regions: {result.num_regions}")
    print(f"Mean gradient norm: {result.mean_gradient_norm:.4f}")
    
    assert result.num_regions >= 1
    
    print("PASSED\n")
    return True


def test_decision_boundary_direction():
    """Test decision boundary direction computation"""
    print("=" * 50)
    print("Test: Decision Boundary Direction")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=30, hidden_dims=[20], output_dim=5)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(30,),
        device=device
    )
    
    torch.manual_seed(42)
    x = torch.randn(1, 30, device=device)
    
    direction = analyzer.find_decision_boundary_direction(x)
    
    print(f"Direction shape: {direction.shape}")
    
    # 检查方向是否已归一化
    norm = torch.norm(direction.view(-1), p=2).item()
    print(f"Direction norm: {norm:.4f}")
    
    assert direction.shape == x.shape
    assert abs(norm - 1.0) < 1e-5, "Direction should be normalized"
    
    print("PASSED\n")
    return True


def test_no_label_analysis():
    """Test analysis without label"""
    print("=" * 50)
    print("Test: Analysis without Label")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=30, hidden_dims=[20], output_dim=5)
    device = get_device()
    
    analyzer = SimpleLinearRegionAnalyzer(
        model=model,
        input_shape=(30,),
        device=device
    )
    
    torch.manual_seed(42)
    x = torch.randn(1, 30, device=device)
    
    # 不提供 label
    result = analyzer.analyze(x, label=None, max_distance=1.0)
    
    print(f"Number of regions: {result.num_regions}")
    print(f"Mean gradient norm: {result.mean_gradient_norm:.4f}")
    print(f"Mean loss change: {result.mean_loss_change:.4f}")
    
    assert result.num_regions >= 1
    assert result.mean_loss_change == 0.0, "Loss change should be 0 when no label provided"
    
    print("PASSED\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Running Tests for SimpleLinearRegionAnalyzer")
    print("=" * 70 + "\n")
    
    tests = [
        ("Creation", test_simple_analyzer_creation),
        ("Gradient Norm", test_gradient_norm_computation),
        ("Activation Pattern", test_activation_pattern),
        ("Boundary Detection", test_boundary_detection),
        ("Single Analysis", test_single_sample_analysis),
        ("Direction Analysis", test_direction_analysis),
        ("Batch Analysis", test_batch_analysis),
        ("CNN Model", test_cnn_model),
        ("Decision Boundary Direction", test_decision_boundary_direction),
        ("No Label Analysis", test_no_label_analysis),
    ]
    
    passed = 0
    failed = 0
    
    for name, test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
