# -*- coding: utf-8 -*-
"""
使用 LinearRegionAnalyzer 分析 ResNet18-CIFAR10 模型
"""

import torch
import sys
import os
import numpy as np
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path. abspath(__file__)))

from torchvision import datasets, transforms
from model import ResNet18
from my. core.analyzer import LinearRegionAnalyzer


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的 ResNet18 模型"""
    # 创建模型（根据你的配置调整参数）
    model = ResNet18(input_channel=3, num_classed=10, BN=False, width=16)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 处理不同的 checkpoint 格式
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def load_cifar10_test(data_dir='data', batch_size=32):
    """加载 CIFAR10 测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets. CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )
    
    test_loader = torch. utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return test_loader, test_dataset


def analyze_single_sample(analyzer, x, label, max_distance=10.0, max_regions=50):
    """分析单个样本"""
    result = analyzer.analyze(
        x=x,
        label=label,
        max_distance=max_distance,
        max_regions=max_regions,
        keep_details=True
    )
    
    print("\n" + "=" * 50)
    print("单样本分析结果")
    print("=" * 50)
    print("预测类别: %d, 真实标签: %d" % (result. predicted_class, label))
    print("第二可能类别: %d" % result.second_class)
    print("Margin (top1 - top2): %.4f" % result.margin)
    print("-" * 50)
    print("经过区域数: %d" % result.num_regions)
    print("遍历距离: %.4f" % result.total_distance)
    print("-" * 50)
    print("平均雅可比范数: %.4f" % result.mean_jacobian_norm)
    print("最大雅可比范数: %.4f" % result.max_jacobian_norm)
    print("边界处平均雅可比变化: %.4f" % result.mean_jacobian_diff)
    print("边界处最大雅可比变化: %.4f" % result.max_jacobian_diff)
    print("-" * 50)
    print("边界处平均 Loss 变化: %.4f" % result.mean_loss_diff)
    print("总 Loss 变化: %.4f" % result.total_loss_change)
    print("-" * 50)
    print("是否跨越决策边界: %s" % result.crossed_decision_boundary)
    print("终点预测类别: %d" % result. final_class)
    
    return result


def analyze_batch_samples(analyzer, data_loader, num_samples=100, max_distance=10.0, max_regions=50):
    """批量分析样本"""
    device = analyzer.device
    
    all_results = []
    samples_analyzed = 0
    
    print("\n" + "=" * 50)
    print("批量分析 (共 %d 个样本)" % num_samples)
    print("=" * 50)
    
    for batch_idx, (data, target) in enumerate(data_loader):
        if samples_analyzed >= num_samples:
            break
        
        # 计算本批次要分析的样本数
        remaining = num_samples - samples_analyzed
        batch_size = min(data.shape[0], remaining)
        
        data = data[:batch_size]. to(device)
        target = target[:batch_size].to(device)
        
        # 批量分析
        batch_result = analyzer.analyze_batch(
            x=data,
            labels=target,
            max_distance=max_distance,
            max_regions=max_regions
        )
        
        all_results. extend(batch_result. results)
        samples_analyzed += batch_size
        
        print("已分析 %d/%d 样本..." % (samples_analyzed, num_samples))
    
    # 汇总统计
    num_correct = sum(1 for r in all_results if r.predicted_class == r. final_class or not r.crossed_decision_boundary)
    num_crossed = sum(1 for r in all_results if r.crossed_decision_boundary)
    
    mean_regions = sum(r. num_regions for r in all_results) / len(all_results)
    mean_jacobian = sum(r.mean_jacobian_norm for r in all_results) / len(all_results)
    mean_jacobian_diff = sum(r.mean_jacobian_diff for r in all_results) / len(all_results)
    mean_loss_diff = sum(r.mean_loss_diff for r in all_results) / len(all_results)
    
    print("\n" + "=" * 50)
    print("批量分析汇总")
    print("=" * 50)
    print("总样本数: %d" % len(all_results))
    print("跨越决策边界的样本数: %d (%.2f%%)" % (num_crossed, 100 * num_crossed / len(all_results)))
    print("-" * 50)
    print("平均经过区域数: %.2f" % mean_regions)
    print("平均雅可比范数: %.4f" % mean_jacobian)
    print("边界处平均雅可比变化: %.4f" % mean_jacobian_diff)
    print("边界处平均 Loss 变化: %.4f" % mean_loss_diff)
    
    return all_results


def compare_boundary_vs_random(analyzer, x, label, num_random=5):
    """比较决策边界方向与随机方向"""
    comparison = analyzer.compare_directions(
        x=x,
        label=label,
        max_distance=10.0,
        max_regions=50,
        num_random=num_random
    )
    
    print("\n" + "=" * 50)
    print("方向比较: 决策边界 vs 随机")
    print("=" * 50)
    
    print("\n决策边界方向:")
    print("  区域数: %d" % comparison['boundary_direction']['num_regions'])
    print("  平均雅可比范数: %.4f" % comparison['boundary_direction']['mean_jacobian_norm'])
    print("  边界处雅可比变化: %.4f" % comparison['boundary_direction']['mean_jacobian_diff'])
    print("  跨越决策边界: %s" % comparison['boundary_direction']['crossed_boundary'])
    
    print("\n随机方向 (平均 %d 次):" % num_random)
    print("  区域数: %.2f ± %.2f" % (
        comparison['random_directions']['num_regions_mean'],
        comparison['random_directions']['num_regions_std']))
    print("  平均雅可比范数: %.4f" % comparison['random_directions']['mean_jacobian_norm'])
    print("  边界处雅可比变化: %.4f" % comparison['random_directions']['mean_jacobian_diff'])
    print("  跨越决策边界比例: %.2f" % comparison['random_directions']['crossing_rate'])
    
    return comparison


def analyze_full_testset(analyzer, data_loader, test_dataset, max_distance=10.0, max_regions=50):
    """对整个测试集进行详细分析"""
    device = analyzer.device
    
    # 记录开始时间
    start_time = time.time()
    
    all_results = []
    all_labels = []
    samples_analyzed = 0
    total_samples = len(test_dataset)
    
    print("\n" + "=" * 80)
    print("全测试集分析 (共 %d 个样本)" % total_samples)
    print("=" * 80)
    
    # 遍历所有批次
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        
        # 批量分析
        batch_result = analyzer.analyze_batch(
            x=data,
            labels=target,
            max_distance=max_distance,
            max_regions=max_regions
        )
        
        all_results.extend(batch_result.results)
        all_labels.extend(target.cpu().numpy().tolist())
        samples_analyzed += data.shape[0]
        
        # 每处理一定数量的样本输出进度
        if (batch_idx + 1) % 10 == 0 or samples_analyzed == total_samples:
            print("进度: %d/%d 样本 (%.2f%%)..." % (
                samples_analyzed, total_samples, 100.0 * samples_analyzed / total_samples))
    
    # 转换为 numpy 数组便于统计
    num_regions = np.array([r.num_regions for r in all_results])
    total_distances = np.array([r.total_distance for r in all_results])
    mean_jacobian_norms = np.array([r.mean_jacobian_norm for r in all_results])
    max_jacobian_norms = np.array([r.max_jacobian_norm for r in all_results])
    mean_jacobian_diffs = np.array([r.mean_jacobian_diff for r in all_results])
    max_jacobian_diffs = np.array([r.max_jacobian_diff for r in all_results])
    mean_loss_diffs = np.array([r.mean_loss_diff for r in all_results])
    total_loss_changes = np.array([r.total_loss_change for r in all_results])
    margins = np.array([r.margin for r in all_results])
    crossed_boundary = np.array([r.crossed_decision_boundary for r in all_results])
    predicted_classes = np.array([r.predicted_class for r in all_results])
    final_classes = np.array([r.final_class for r in all_results])
    all_labels = np.array(all_labels)
    
    # 计算准确率相关统计
    initial_correct = (predicted_classes == all_labels)
    final_correct = (final_classes == all_labels)
    initial_accuracy = initial_correct.mean()
    final_accuracy = final_correct.mean()
    crossing_rate = crossed_boundary.mean()
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印统计报告（只包含均值和标准差）
    print("\n" + "=" * 80)
    print("全测试集分析结果汇总")
    print("=" * 80)
    
    print("\n【统计信息】")
    print("-" * 80)
    print("总样本数: %d" % len(all_results))
    print("初始预测准确率: %.4f%%" % (100 * initial_accuracy))
    print("终点预测准确率: %.4f%%" % (100 * final_accuracy))
    print("跨越决策边界比例: %.4f%%" % (100 * crossing_rate))
    
    print("\n【区域数】")
    print("-" * 80)
    print("均值: %.4f" % num_regions.mean())
    print("标准差: %.4f" % num_regions.std())
    
    print("\n【遍历距离】")
    print("-" * 80)
    print("均值: %.4f" % total_distances.mean())
    print("标准差: %.4f" % total_distances.std())
    
    print("\n【平均雅可比范数】")
    print("-" * 80)
    print("均值: %.4f" % mean_jacobian_norms.mean())
    print("标准差: %.4f" % mean_jacobian_norms.std())
    
    print("\n【最大雅可比范数】")
    print("-" * 80)
    print("均值: %.4f" % max_jacobian_norms.mean())
    print("标准差: %.4f" % max_jacobian_norms.std())
    
    print("\n【边界处平均雅可比变化】")
    print("-" * 80)
    print("均值: %.4f" % mean_jacobian_diffs.mean())
    print("标准差: %.4f" % mean_jacobian_diffs.std())
    
    print("\n【边界处最大雅可比变化】")
    print("-" * 80)
    print("均值: %.4f" % max_jacobian_diffs.mean())
    print("标准差: %.4f" % max_jacobian_diffs.std())
    
    print("\n【边界处平均损失变化】")
    print("-" * 80)
    print("均值: %.4f" % mean_loss_diffs.mean())
    print("标准差: %.4f" % mean_loss_diffs.std())
    
    print("\n【总损失变化 (起点到终点)】")
    print("-" * 80)
    print("均值: %.4f" % total_loss_changes.mean())
    print("标准差: %.4f" % total_loss_changes.std())
    
    print("\n【Margin (top1 - top2 logit 差)】")
    print("-" * 80)
    print("均值: %.4f" % margins.mean())
    print("标准差: %.4f" % margins.std())
    
    print("\n【运行时间】")
    print("-" * 80)
    print("总运行时间: %.2f 秒" % elapsed_time)
    print("平均每个样本: %.4f 秒" % (elapsed_time / len(all_results)))
    
    print("\n" + "=" * 80)
    print("全测试集分析完成！")
    print("=" * 80)
    
    return all_results


def main():
    # ============ 配置 ============
    # 修改为你的模型路径
    checkpoint_path = "/data/tqh/Projects/LR/saved/models/Cifar10_ResNet18_adam/1126_140841/checkpoint-epoch20.pth"
    data_dir = "data"
    device = "cuda:3" if torch. cuda.is_available() else "cpu"
    
    print("使用设备: %s" % device)
    
    # ============ 加载模型和数据 ============
    print("\n加载模型...")
    model = load_model(checkpoint_path, device)
    
    print("加载 CIFAR10 测试集...")
    test_loader, test_dataset = load_cifar10_test(data_dir, batch_size=32)
    
    # ============ 创建分析器 ============
    print("创建 LinearRegionAnalyzer...")
    analyzer = LinearRegionAnalyzer(
        model=model,
        input_shape=(3, 32, 32),  # CIFAR10 图像尺寸
        device=device,
        batch_size=32,
        num_samples_per_region=3
    )
    
    # # ============ 单样本分析 ============
    # # 取第一个测试样本
    # sample_data, sample_label = test_dataset[1]
    # sample_data = sample_data. unsqueeze(0).to(device)
    
    # result = analyze_single_sample(
    #     analyzer, sample_data, sample_label,
    #     max_distance=20.0, max_regions=1000
    # )
    
    # # ============ 方向比较 ============
    # compare_boundary_vs_random(
    #     analyzer, sample_data, sample_label, num_random=5
    # )
    
    # # ============ 批量分析 ============
    # # 分析前 100 个测试样本
    # all_results = analyze_batch_samples(
    #     analyzer, test_loader, 
    #     num_samples=100,
    #     max_distance=10.0, 
    #     max_regions=50
    # )
    
    # ============ 全测试集分析 ============
    print("\n\n" + "=" * 80)
    print("开始全测试集分析")
    print("=" * 80)
    
    full_testset_results = analyze_full_testset(
        analyzer, test_loader, test_dataset,
        max_distance=1.0,
        max_regions=500
    )
    
    # ============ 清理 ============
    analyzer.cleanup()
    print("\n分析完成！")


if __name__ == "__main__":
    main()