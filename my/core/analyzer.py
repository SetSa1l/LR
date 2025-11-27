# -*- coding: utf-8 -*-
"""
LinearRegionAnalyzer: 整合所有组件的高层分析器

提供简洁的 API 完成完整的线性区域分析流程：
1. 计算朝向决策边界的方向
2. 沿该方向遍历线性区域
3. 计算每个区域的性质
4. 返回汇总结果
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field

from .model_wrapper import ModelWrapper, normalize_direction
from .region_traverser import LinearRegionTraverser, TraversalResult, BatchTraversalResult
from .direction_finder import DecisionBoundaryDirectionFinder
from .region_properties import RegionPropertyAnalyzer, TraversalProperties


@dataclass
class AnalysisResult:
    """单个样本的完整分析结果"""
    # 基本信息
    predicted_class: int
    second_class: int
    margin: float  # top1 - top2 logit 差
    
    # 遍历信息
    num_regions: int
    total_distance: float
    
    # 区域性质统计
    mean_jacobian_norm: float
    max_jacobian_norm: float
    mean_jacobian_diff: float  # 边界处的平均雅可比变化
    max_jacobian_diff: float
    mean_loss_diff: float      # 边界处的平均 loss 变化
    total_loss_change: float   # 从起点到终点的总 loss 变化
    
    # 是否跨越了决策边界
    crossed_decision_boundary: bool
    final_class: int
    
    # 详细结果（可选保留）
    traversal: Optional[TraversalResult] = None
    properties: Optional[TraversalProperties] = None


@dataclass
class BatchAnalysisResult:
    """批量分析结果"""
    batch_size: int
    results: List[AnalysisResult] = field(default_factory=list)
    
    # 批量统计
    mean_num_regions: float = 0.0
    mean_jacobian_norm: float = 0.0
    mean_jacobian_diff: float = 0.0
    boundary_crossing_rate: float = 0.0  # 跨越决策边界的比例
    
    def compute_statistics(self):
        """计算批量统计量"""
        if len(self.results) == 0:
            return
        
        self.mean_num_regions = sum(r.num_regions for r in self.results) / len(self.results)
        self.mean_jacobian_norm = sum(r.mean_jacobian_norm for r in self.results) / len(self.results)
        self.mean_jacobian_diff = sum(r.mean_jacobian_diff for r in self.results) / len(self.results)
        self.boundary_crossing_rate = sum(1 for r in self.results if r.crossed_decision_boundary) / len(self.results)


class LinearRegionAnalyzer:
    """
    线性区域分析器（整合所有组件）
    
    用法:
        analyzer = LinearRegionAnalyzer(model, input_shape=(784,), device='cuda')
        
        # 单样本分析
        result = analyzer.analyze(x, label)
        print("经过 %d 个区域" % result.num_regions)
        print("平均雅可比范数: %.4f" % result.mean_jacobian_norm)
        
        # 批量分析
        batch_result = analyzer.analyze_batch(x_batch, labels)
        print("平均区域数: %.2f" % batch_result.mean_num_regions)
        
        # 清理
        analyzer.cleanup()
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        batch_size: int = 128,
        num_samples_per_region: int = 3
    ):
        """
        Args:
            model: PyTorch 模型
            input_shape: 输入形状（不含 batch 维度）
            device: 计算设备
            batch_size: ModelWrapper 的 batch size
            num_samples_per_region: 每个区域内采样点数（用于计算平均 loss）
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device
        
        # 初始化各组件
        self.direction_finder = DecisionBoundaryDirectionFinder(model, device)
        self.wrapper = ModelWrapper(model, input_shape, batch_size, device)
        self.traverser = LinearRegionTraverser(self.wrapper)
        self.property_analyzer = RegionPropertyAnalyzer(model, device, num_samples_per_region)
        
        # 获取模型 dtype
        self._model_dtype = next(model.parameters()).dtype
    
    def _get_final_class(self, x: torch.Tensor, direction: torch.Tensor, 
                        traversal, predicted_class: int) -> int:
        """统一的终点类别检查 - 减少重复代码"""
        if traversal.num_regions == 0:
            return predicted_class
        
        last_region = traversal.regions[-1]
        end_point = x + last_region.exit_t * direction
        with torch.no_grad():
            return self.model(end_point).argmax(dim=1).item()
    
    def analyze(
        self,
        x: torch.Tensor,
        label: Optional[Union[int, torch.Tensor]] = None,
        max_distance: float = 10.0,
        max_regions: int = 100,
        keep_details: bool = False
    ) -> AnalysisResult:
        """
        分析单个样本
        
        Args:
            x: 输入数据 (1, *input_shape) 或 (*input_shape)
            label: 真实标签（用于计算 loss）
            max_distance: 最大遍历距离
            max_regions: 最大区域数
            keep_details: 是否保留详细的遍历和性质结果
        
        Returns:
            AnalysisResult
        """
        # 确保输入形状正确
        if x.dim() == len(self.input_shape):
            x = x.unsqueeze(0)
        
        x = x.to(device=self.device, dtype=self._model_dtype)
        
        # 1. 获取方向和预测信息
        direction, top1, top2, margin = self.direction_finder.find_direction_with_info(x)
        predicted_class = top1.item()
        second_class = top2.item()
        margin_val = margin.item()
        
        # 2. 遍历线性区域
        traversal = self.traverser.traverse(
            start=x,
            direction=direction,
            max_distance=max_distance,
            max_regions=max_regions,
            normalize_dir=False  # 方向已归一化
        )
        
        # 3. 分析区域性质
        properties = self.property_analyzer.analyze_traversal(
            traversal, x, direction, label
        )
        
        # 4. 检查是否跨越决策边界
        final_class = self._get_final_class(x, direction, traversal, predicted_class)
        crossed = (final_class != predicted_class)
        
        return AnalysisResult(
            predicted_class=predicted_class,
            second_class=second_class,
            margin=margin_val,
            num_regions=properties.num_regions,
            total_distance=properties.total_distance,
            mean_jacobian_norm=properties.mean_jacobian_norm,
            max_jacobian_norm=properties.max_jacobian_norm,
            mean_jacobian_diff=properties.mean_jacobian_diff,
            max_jacobian_diff=properties.max_jacobian_diff,
            mean_loss_diff=properties.mean_loss_diff,
            total_loss_change=properties.total_loss_change,
            crossed_decision_boundary=crossed,
            final_class=final_class,
            traversal=traversal if keep_details else None,
            properties=properties if keep_details else None
        )
    
    def analyze_batch(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        max_distance: float = 10.0,
        max_regions: int = 100,
        keep_details: bool = False
    ) -> BatchAnalysisResult:
        """
        批量分析
        
        Args:
            x: 输入数据 (batch, *input_shape)
            labels: 真实标签 (batch,)
            max_distance: 最大遍历距离
            max_regions: 最大区域数
            keep_details: 是否保留详细结果
        
        Returns:
            BatchAnalysisResult
        """
        batch_size = x.shape[0]
        x = x.to(device=self.device, dtype=self._model_dtype)
        
        # 1. 批量计算方向
        directions, top1_indices, top2_indices, margins = \
            self.direction_finder.find_direction_with_info(x)
        
        # 2.  批量遍历
        batch_traversal = self.traverser.traverse_batch(
            starts=x,
            directions=directions,
            max_distance=max_distance,
            max_regions=max_regions,
            normalize_dir=False
        )
        
        # 3. 批量分析性质
        all_properties = self.property_analyzer.analyze_batch(
            batch_traversal, x, directions, labels
        )
        
        # 4. 构建结果
        results = []
        for i in range(batch_size):
            # 获取单个样本的遍历结果
            traversal = batch_traversal.get_single_result(i, x, directions)
            properties = all_properties[i]
            
            # 检查是否跨越决策边界
            predicted_class = top1_indices[i].item()
            final_class = self._get_final_class(
                x[i:i+1], directions[i:i+1], traversal, predicted_class
            )
            crossed = (final_class != predicted_class)
            
            result = AnalysisResult(
                predicted_class=predicted_class,
                second_class=top2_indices[i].item(),
                margin=margins[i].item(),
                num_regions=properties.num_regions,
                total_distance=properties.total_distance,
                mean_jacobian_norm=properties.mean_jacobian_norm,
                max_jacobian_norm=properties.max_jacobian_norm,
                mean_jacobian_diff=properties.mean_jacobian_diff,
                max_jacobian_diff=properties.max_jacobian_diff,
                mean_loss_diff=properties.mean_loss_diff,
                total_loss_change=properties.total_loss_change,
                crossed_decision_boundary=crossed,
                final_class=final_class,
                traversal=traversal if keep_details else None,
                properties=properties if keep_details else None
            )
            results.append(result)
        
        batch_result = BatchAnalysisResult(
            batch_size=batch_size,
            results=results
        )
        batch_result.compute_statistics()
        
        return batch_result
    
    def analyze_with_random_direction(
        self,
        x: torch.Tensor,
        label: Optional[Union[int, torch.Tensor]] = None,
        max_distance: float = 10.0,
        max_regions: int = 100,
        seed: Optional[int] = None
    ) -> AnalysisResult:
        """
        使用随机方向分析（用于对比实验）
        
        Args:
            x: 输入数据
            label: 标签
            max_distance: 最大距离
            max_regions: 最大区域数
            seed: 随机种子
        
        Returns:
            AnalysisResult
        """
        if x.dim() == len(self.input_shape):
            x = x.unsqueeze(0)
        
        x = x.to(device=self.device, dtype=self._model_dtype)
        
        # 随机方向
        if seed is not None:
            torch.manual_seed(seed)
        direction = torch.randn_like(x)
        direction = normalize_direction(direction)
        
        # 获取预测信息
        with torch.no_grad():
            logits = self.model(x)
            top2_vals, top2_idx = torch.topk(logits, k=2, dim=1)
        
        predicted_class = top2_idx[0, 0].item()
        second_class = top2_idx[0, 1].item()
        margin_val = (top2_vals[0, 0] - top2_vals[0, 1]).item()
        
        # 遍历
        traversal = self.traverser.traverse(
            start=x, direction=direction,
            max_distance=max_distance, max_regions=max_regions,
            normalize_dir=False
        )
        
        # 分析
        properties = self.property_analyzer.analyze_traversal(
            traversal, x, direction, label
        )
        
        # 检查终点
        final_class = self._get_final_class(x, direction, traversal, predicted_class)
        
        return AnalysisResult(
            predicted_class=predicted_class,
            second_class=second_class,
            margin=margin_val,
            num_regions=properties.num_regions,
            total_distance=properties.total_distance,
            mean_jacobian_norm=properties.mean_jacobian_norm,
            max_jacobian_norm=properties.max_jacobian_norm,
            mean_jacobian_diff=properties.mean_jacobian_diff,
            max_jacobian_diff=properties.max_jacobian_diff,
            mean_loss_diff=properties.mean_loss_diff,
            total_loss_change=properties.total_loss_change,
            crossed_decision_boundary=(final_class != predicted_class),
            final_class=final_class
        )
    
    def compare_directions(
        self,
        x: torch.Tensor,
        label: Optional[Union[int, torch.Tensor]] = None,
        max_distance: float = 10.0,
        max_regions: int = 100,
        num_random: int = 5
    ) -> dict:
        """
        比较决策边界方向和随机方向的分析结果
        
        Args:
            x: 输入数据
            label: 标签
            max_distance: 最大距离
            max_regions: 最大区域数
            num_random: 随机方向数量
        
        Returns:
            包含比较结果的字典
        """
        # 决策边界方向
        boundary_result = self.analyze(x, label, max_distance, max_regions)
        
        # 多个随机方向
        random_results = []
        for i in range(num_random):
            result = self.analyze_with_random_direction(
                x, label, max_distance, max_regions, seed=i
            )
            random_results.append(result)
        
        # 统计
        random_num_regions = [r.num_regions for r in random_results]
        random_jacobian_norms = [r.mean_jacobian_norm for r in random_results]
        random_jacobian_diffs = [r.mean_jacobian_diff for r in random_results]
        random_crossing = [r.crossed_decision_boundary for r in random_results]
        
        return {
            'boundary_direction': {
                'num_regions': boundary_result.num_regions,
                'mean_jacobian_norm': boundary_result.mean_jacobian_norm,
                'mean_jacobian_diff': boundary_result.mean_jacobian_diff,
                'crossed_boundary': boundary_result.crossed_decision_boundary
            },
            'random_directions': {
                'num_regions_mean': sum(random_num_regions) / len(random_num_regions),
                'num_regions_std': torch.tensor(random_num_regions).float().std().item(),
                'mean_jacobian_norm': sum(random_jacobian_norms) / len(random_jacobian_norms),
                'mean_jacobian_diff': sum(random_jacobian_diffs) / len(random_jacobian_diffs),
                'crossing_rate': sum(random_crossing) / len(random_crossing)
            }
        }
    
    def cleanup(self):
        """清理资源"""
        self.wrapper.cleanup()