# -*- coding: utf-8 -*-
"""
使用 LinearRegionAnalyzer 分析 ResNet18-CIFAR10 模型
"""

import torch
import sys
import os
import numpy as np
from collections import defaultdict

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
    
    # 按类别统计
    class_stats = defaultdict(lambda: {
        'count': 0,
        'num_regions': [],
        'mean_jacobian_norm': [],
        'crossed_boundary': [],
        'initial_correct': [],
        'final_correct': []
    })
    
    for i, label in enumerate(all_labels):
        stats = class_stats[int(label)]
        stats['count'] += 1
        stats['num_regions'].append(all_results[i].num_regions)
        stats['mean_jacobian_norm'].append(all_results[i].mean_jacobian_norm)
        stats['crossed_boundary'].append(all_results[i].crossed_decision_boundary)
        stats['initial_correct'].append(predicted_classes[i] == label)
        stats['final_correct'].append(final_classes[i] == label)
    
    # 打印详细统计报告
    print("\n" + "=" * 80)
    print("全测试集分析结果汇总")
    print("=" * 80)
    
    # 1. 基本信息
    print("\n【基本信息】")
    print("-" * 80)
    print("总样本数: %d" % len(all_results))
    print("初始预测准确率: %.4f%% (%d/%d)" % (
        100 * initial_accuracy, initial_correct.sum(), len(all_results)))
    print("终点预测准确率: %.4f%% (%d/%d)" % (
        100 * final_accuracy, final_correct.sum(), len(all_results)))
    print("准确率变化: %.4f%%" % (100 * (final_accuracy - initial_accuracy)))
    
    # 2. 决策边界跨越统计
    print("\n【决策边界跨越统计】")
    print("-" * 80)
    num_crossed = crossed_boundary.sum()
    crossing_rate = crossed_boundary.mean()
    print("跨越决策边界的样本数: %d (%.4f%%)" % (num_crossed, 100 * crossing_rate))
    print("未跨越决策边界的样本数: %d (%.4f%%)" % (
        len(all_results) - num_crossed, 100 * (1 - crossing_rate)))
    
    # 跨越边界 vs 未跨越边界的统计对比
    crossed_mask = crossed_boundary
    not_crossed_mask = ~crossed_boundary
    
    if crossed_mask.sum() > 0:
        print("\n跨越边界的样本统计:")
        print("  平均区域数: %.2f (未跨越: %.2f)" % (
            num_regions[crossed_mask].mean(),
            num_regions[not_crossed_mask].mean() if not_crossed_mask.sum() > 0 else 0))
        print("  平均雅可比范数: %.4f (未跨越: %.4f)" % (
            mean_jacobian_norms[crossed_mask].mean(),
            mean_jacobian_norms[not_crossed_mask].mean() if not_crossed_mask.sum() > 0 else 0))
        print("  平均损失变化: %.4f (未跨越: %.4f)" % (
            total_loss_changes[crossed_mask].mean(),
            total_loss_changes[not_crossed_mask].mean() if not_crossed_mask.sum() > 0 else 0))
    
    # 3. 区域数统计
    print("\n【区域数统计】")
    print("-" * 80)
    print("均值: %.2f" % num_regions.mean())
    print("中位数: %.2f" % np.median(num_regions))
    print("标准差: %.2f" % num_regions.std())
    print("最小值: %d" % num_regions.min())
    print("最大值: %d" % num_regions.max())
    print("25%%分位数: %.2f" % np.percentile(num_regions, 25))
    print("75%%分位数: %.2f" % np.percentile(num_regions, 75))
    print("90%%分位数: %.2f" % np.percentile(num_regions, 90))
    print("95%%分位数: %.2f" % np.percentile(num_regions, 95))
    print("99%%分位数: %.2f" % np.percentile(num_regions, 99))
    
    # 4. 遍历距离统计
    print("\n【遍历距离统计】")
    print("-" * 80)
    print("均值: %.4f" % total_distances.mean())
    print("中位数: %.4f" % np.median(total_distances))
    print("标准差: %.4f" % total_distances.std())
    print("最小值: %.4f" % total_distances.min())
    print("最大值: %.4f" % total_distances.max())
    
    # 5. 雅可比范数统计
    print("\n【雅可比范数统计】")
    print("-" * 80)
    print("平均雅可比范数:")
    print("  均值: %.4f" % mean_jacobian_norms.mean())
    print("  中位数: %.4f" % np.median(mean_jacobian_norms))
    print("  标准差: %.4f" % mean_jacobian_norms.std())
    print("  最小值: %.4f" % mean_jacobian_norms.min())
    print("  最大值: %.4f" % mean_jacobian_norms.max())
    
    print("\n最大雅可比范数:")
    print("  均值: %.4f" % max_jacobian_norms.mean())
    print("  中位数: %.4f" % np.median(max_jacobian_norms))
    print("  标准差: %.4f" % max_jacobian_norms.std())
    print("  最大值: %.4f" % max_jacobian_norms.max())
    
    # 6. 边界处雅可比变化统计
    print("\n【边界处雅可比变化统计】")
    print("-" * 80)
    print("平均变化:")
    print("  均值: %.4f" % mean_jacobian_diffs.mean())
    print("  中位数: %.4f" % np.median(mean_jacobian_diffs))
    print("  标准差: %.4f" % mean_jacobian_diffs.std())
    print("  最大值: %.4f" % mean_jacobian_diffs.max())
    
    print("\n最大变化:")
    print("  均值: %.4f" % max_jacobian_diffs.mean())
    print("  中位数: %.4f" % np.median(max_jacobian_diffs))
    print("  标准差: %.4f" % max_jacobian_diffs.std())
    print("  最大值: %.4f" % max_jacobian_diffs.max())
    
    # 7. 损失变化统计
    print("\n【损失变化统计】")
    print("-" * 80)
    print("边界处平均损失变化:")
    print("  均值: %.4f" % mean_loss_diffs.mean())
    print("  中位数: %.4f" % np.median(mean_loss_diffs))
    print("  标准差: %.4f" % mean_loss_diffs.std())
    print("  最小值: %.4f" % mean_loss_diffs.min())
    print("  最大值: %.4f" % mean_loss_diffs.max())
    
    print("\n总损失变化 (起点到终点):")
    print("  均值: %.4f" % total_loss_changes.mean())
    print("  中位数: %.4f" % np.median(total_loss_changes))
    print("  标准差: %.4f" % total_loss_changes.std())
    print("  最小值: %.4f" % total_loss_changes.min())
    print("  最大值: %.4f" % total_loss_changes.max())
    
    # 8. Margin 统计
    print("\n【Margin 统计 (top1 - top2 logit 差)】")
    print("-" * 80)
    print("均值: %.4f" % margins.mean())
    print("中位数: %.4f" % np.median(margins))
    print("标准差: %.4f" % margins.std())
    print("最小值: %.4f" % margins.min())
    print("最大值: %.4f" % margins.max())
    
    # 9. 按类别统计
    print("\n【按类别统计】")
    print("-" * 80)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    for class_idx in sorted(class_stats.keys()):
        stats = class_stats[class_idx]
        class_name = cifar10_classes[class_idx] if class_idx < len(cifar10_classes) else str(class_idx)
        
        regions_arr = np.array(stats['num_regions'])
        jacobian_arr = np.array(stats['mean_jacobian_norm'])
        crossed_arr = np.array(stats['crossed_boundary'])
        init_correct_arr = np.array(stats['initial_correct'])
        final_correct_arr = np.array(stats['final_correct'])
        
        print("\n类别 %d (%s):" % (class_idx, class_name))
        print("  样本数: %d" % stats['count'])
        print("  初始准确率: %.2f%% (%d/%d)" % (
            100 * init_correct_arr.mean(), init_correct_arr.sum(), stats['count']))
        print("  终点准确率: %.2f%% (%d/%d)" % (
            100 * final_correct_arr.mean(), final_correct_arr.sum(), stats['count']))
        print("  跨越边界比例: %.2f%% (%d/%d)" % (
            100 * crossed_arr.mean(), crossed_arr.sum(), stats['count']))
        print("  平均区域数: %.2f" % regions_arr.mean())
        print("  平均雅可比范数: %.4f" % jacobian_arr.mean())
    
    # 10. 预测正确 vs 错误的样本对比
    print("\n【预测正确 vs 错误样本对比】")
    print("-" * 80)
    
    correct_mask = initial_correct
    wrong_mask = ~initial_correct
    
    if wrong_mask.sum() > 0:
        print("初始预测正确的样本 (%d 个):" % correct_mask.sum())
        print("  平均区域数: %.2f" % num_regions[correct_mask].mean())
        print("  平均雅可比范数: %.4f" % mean_jacobian_norms[correct_mask].mean())
        print("  跨越边界比例: %.2f%%" % (100 * crossed_boundary[correct_mask].mean()))
        print("  平均 Margin: %.4f" % margins[correct_mask].mean())
        
        print("\n初始预测错误的样本 (%d 个):" % wrong_mask.sum())
        print("  平均区域数: %.2f" % num_regions[wrong_mask].mean())
        print("  平均雅可比范数: %.4f" % mean_jacobian_norms[wrong_mask].mean())
        print("  跨越边界比例: %.2f%%" % (100 * crossed_boundary[wrong_mask].mean()))
        print("  平均 Margin: %.4f" % margins[wrong_mask].mean())
    
    # 11. 区域数分布区间统计
    print("\n【区域数分布区间统计】")
    print("-" * 80)
    bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, float('inf')]
    bin_labels = ['0', '1', '2-5', '6-10', '11-20', '21-50', '51-100', '101-200', '201-500', '500+']
    
    hist, _ = np.histogram(num_regions, bins=bins)
    for i, (label, count) in enumerate(zip(bin_labels, hist)):
        if count > 0:
            print("  %s 个区域: %d 样本 (%.2f%%)" % (
                label, count, 100 * count / len(all_results)))
    
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
        batch_size=64,
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
        max_distance=10.0,
        max_regions=50
    )
    
    # ============ 清理 ============
    analyzer.cleanup()
    print("\n分析完成！")


if __name__ == "__main__":
    main()