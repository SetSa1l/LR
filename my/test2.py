# -*- coding: utf-8 -*-
"""
Phase 2 测试脚本：决策边界方向计算
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
    find_decision_boundary_direction,
    normalize_direction
)


def create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers. append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn. Linear(prev_dim, output_dim))
    return nn. Sequential(*layers)


def test_direction_finder_basic():
    """测试基本的方向计算"""
    print("=" * 50)
    print("Test 1: Basic Direction Finding")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    x = torch.randn(4, 100, device=device)
    directions = finder.find_direction(x)
    
    print("Input shape: %s" % str(x.shape))
    print("Direction shape: %s" % str(directions. shape))
    
    # 检查方向是否归一化
    norms = torch.norm(directions. view(4, -1), p=2, dim=1)
    print("Direction norms: %s" % norms)
    
    assert directions.shape == x.shape, "Direction shape should match input shape"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Directions should be normalized"
    
    print("PASSED\n")
    return True


def test_direction_with_info():
    """测试带信息的方向计算"""
    print("=" * 50)
    print("Test 2: Direction with Info")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch. cuda.is_available() else "cpu"
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    x = torch.randn(4, 100, device=device)
    directions, top1_idx, top2_idx, margins = finder. find_direction_with_info(x)
    
    print("Top1 classes: %s" % top1_idx)
    print("Top2 classes: %s" % top2_idx)
    print("Margins (top1 - top2): %s" % margins)
    
    assert directions.shape == x. shape
    assert top1_idx. shape == (4,)
    assert top2_idx.shape == (4,)
    assert margins. shape == (4,)
    
    # top1 和 top2 应该不同
    assert torch.all(top1_idx != top2_idx), "Top1 and Top2 should be different"
    
    # margin 应该是正的（top1 > top2）
    assert torch.all(margins >= 0), "Margins should be non-negative"
    
    print("PASSED\n")
    return True


def test_direction_moves_toward_boundary():
    """测试方向是否真的朝向决策边界"""
    print("=" * 50)
    print("Test 3: Direction Moves Toward Boundary")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch.cuda. is_available() else "cpu"
    model.to(device)
    model.eval()
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    torch.manual_seed(42)
    x = torch.randn(1, 100, device=device)
    direction = finder.find_direction(x)
    
    # 计算原始 margin
    with torch.no_grad():
        logits_original = model(x)
        top2_vals, top2_idx = torch.topk(logits_original, k=2, dim=1)
        margin_original = (top2_vals[0, 0] - top2_vals[0, 1]).item()
    
    print("Original margin (top1 - top2): %.4f" % margin_original)
    print("Top1 class: %d, Top2 class: %d" % (top2_idx[0, 0]. item(), top2_idx[0, 1].item()))
    
    # 沿方向移动
    step_sizes = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    print("\nMoving along direction:")
    
    for step in step_sizes:
        x_moved = x + step * direction
        with torch.no_grad():
            logits_moved = model(x_moved)
            top2_vals_moved, _ = torch.topk(logits_moved, k=2, dim=1)
            margin_moved = (top2_vals_moved[0, 0] - top2_vals_moved[0, 1]).item()
        
        print("  step=%.1f: margin=%.4f (reduction=%.4f)" % (
            step, margin_moved, margin_original - margin_moved))
    
    # 移动后 margin 应该减小
    x_moved = x + 0.5 * direction
    with torch.no_grad():
        logits_moved = model(x_moved)
        top2_vals_moved, _ = torch.topk(logits_moved, k=2, dim=1)
        margin_moved = (top2_vals_moved[0, 0] - top2_vals_moved[0, 1]).item()
    
    assert margin_moved < margin_original, "Moving along direction should reduce margin"
    
    print("PASSED\n")
    return True


def test_direction_to_target_class():
    """测试朝向特定类别的方向"""
    print("=" * 50)
    print("Test 4: Direction to Target Class")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model. to(device)
    model.eval()
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    torch.manual_seed(123)
    x = torch.randn(1, 100, device=device)
    
    with torch.no_grad():
        logits = model(x)
        current_class = logits.argmax(dim=1).item()
    
    # 选择一个不同的目标类别
    target_class = (current_class + 1) % 10
    
    print("Current predicted class: %d" % current_class)
    print("Target class: %d" % target_class)
    
    direction = finder.find_direction_to_target_class(x, target_class)
    
    # 沿方向移动后，目标类别的 logit 应该增加
    with torch.no_grad():
        original_target_logit = logits[0, target_class]. item()
        
        x_moved = x + 1.0 * direction
        logits_moved = model(x_moved)
        moved_target_logit = logits_moved[0, target_class].item()
    
    print("Original target logit: %.4f" % original_target_logit)
    print("After moving, target logit: %.4f" % moved_target_logit)
    print("Increase: %.4f" % (moved_target_logit - original_target_logit))
    
    assert moved_target_logit > original_target_logit, "Target logit should increase"
    
    print("PASSED\n")
    return True


def test_adversarial_direction():
    """测试对抗方向"""
    print("=" * 50)
    print("Test 5: Adversarial Direction")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch.cuda. is_available() else "cpu"
    model.to(device)
    model. eval()
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    torch.manual_seed(456)
    x = torch. randn(4, 100, device=device)
    
    with torch. no_grad():
        logits = model(x)
        labels = logits.argmax(dim=1)  # 使用预测作为"正确"标签
    
    direction = finder.find_adversarial_direction(x, labels)
    
    print("Labels: %s" % labels)
    
    # 沿对抗方向移动后，正确类别的 logit 应该下降
    with torch.no_grad():
        batch_idx = torch.arange(4, device=device)
        original_correct_logits = logits[batch_idx, labels]
        
        x_moved = x + 1.0 * direction
        logits_moved = model(x_moved)
        moved_correct_logits = logits_moved[batch_idx, labels]
    
    print("Original correct logits: %s" % original_correct_logits)
    print("After moving: %s" % moved_correct_logits)
    
    # 大多数样本的正确 logit 应该下降
    decreased = (moved_correct_logits < original_correct_logits).sum().item()
    print("Samples with decreased correct logit: %d/4" % decreased)
    
    assert decreased >= 3, "Most samples should have decreased correct logit"
    
    print("PASSED\n")
    return True


def test_batch_directions():
    """测试批量方向计算"""
    print("=" * 50)
    print("Test 6: Batch Direction Computation")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    batch_size = 16
    x = torch.randn(batch_size, 100, device=device)
    
    directions = finder. find_direction(x)
    
    print("Batch size: %d" % batch_size)
    print("Directions shape: %s" % str(directions. shape))
    
    # 每个方向都应该归一化
    norms = torch.norm(directions.view(batch_size, -1), p=2, dim=1)
    print("All norms close to 1: %s" % torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
    
    assert directions.shape == (batch_size, 100)
    assert torch. allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    print("PASSED\n")
    return True


def test_integration_with_traverser():
    """测试与 LinearRegionTraverser 的集成"""
    print("=" * 50)
    print("Test 7: Integration with Traverser")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, hidden_dims=[50, 30], output_dim=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建方向查找器
    finder = DecisionBoundaryDirectionFinder(model, device)
    
    # 创建模型包装器和遍历器
    wrapper = ModelWrapper(model, input_shape=(100,), batch_size=32, device=device)
    traverser = LinearRegionTraverser(wrapper)
    
    # 测试数据
    torch.manual_seed(789)
    x = torch.randn(1, 100, device=device)
    
    # 计算朝向决策边界的方向
    direction, top1, top2, margin = finder.find_direction_with_info(x)
    
    print("Initial prediction: class %d (margin to class %d: %.4f)" % (
        top1.item(), top2.item(), margin. item()))
    
    # 沿该方向遍历线性区域
    result = traverser.traverse(
        start=x,
        direction=direction,
        max_distance=5.0,
        max_regions=30,
        normalize_dir=False  # 方向已归一化
    )
    
    print("Traversed %d regions" % result.num_regions)
    print("Total distance: %.4f" % result.total_distance)
    
    # 检查最终预测是否改变
    if result.num_regions > 0:
        last_region = result.regions[-1]
        end_point = x + last_region.exit_t * direction
        
        with torch.no_grad():
            end_logits = model(end_point. to(device))
            end_class = end_logits.argmax(dim=1).item()
            end_top2 = torch.topk(end_logits, k=2, dim=1)
            end_margin = (end_top2. values[0, 0] - end_top2.values[0, 1]).item()
        
        print("End prediction: class %d (margin: %.4f)" % (end_class, end_margin))
        
        if end_class != top1.item():
            print("*** Decision boundary crossed! ***")
    
    wrapper.cleanup()
    print("PASSED\n")
    return True


def test_convenience_function():
    """测试便捷函数"""
    print("=" * 50)
    print("Test 8: Convenience Function")
    print("=" * 50)
    
    model = create_simple_mlp(input_dim=100, output_dim=10)
    device = "cuda" if torch. cuda.is_available() else "cpu"
    
    x = torch.randn(3, 100, device=device)
    direction = find_decision_boundary_direction(model, x, device)
    
    print("Direction shape: %s" % str(direction.shape))
    assert direction.shape == x.shape
    
    print("PASSED\n")
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("Running Phase 2 Tests: Decision Boundary Direction Finder")
    print("=" * 60 + "\n")
    
    tests = [
        test_direction_finder_basic,
        test_direction_with_info,
        test_direction_moves_toward_boundary,
        test_direction_to_target_class,
        test_adversarial_direction,
        test_batch_directions,
        test_integration_with_traverser,
        test_convenience_function,
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