# -*- coding: utf-8 -*-
"""
精简的线性区域分析器 - 只计算4个核心指标

核心指标：
1. num_regions: 沿着指定方向的线性区域数量
2. mean_gradient_norm: 平均梯度范数值
3. mean_gradient_norm_change: 平均梯度范数变化值（边界处）
4. mean_loss_change: 平均损失变化值（边界处）

设计原则：
- 最小化内存占用（不存储完整激活模式历史）
- 只计算需要的指标（删除谱范数、logit变化等）
- 及时释放中间tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from copy import deepcopy
import numpy as np


# 常量
EPSILON = 1e-6


@dataclass
class SimpleAnalysisResult:
    """精简的分析结果 - 只包含4个核心指标"""
    num_regions: int                    # 区域数量
    mean_gradient_norm: float           # 平均梯度范数
    mean_gradient_norm_change: float    # 平均梯度范数变化
    mean_loss_change: float             # 平均损失变化


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    """归一化方向向量"""
    flat = direction.view(direction.shape[0], -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    norm = norm.view((direction.shape[0],) + (1,) * (len(direction.shape) - 1))
    return direction / (norm + 1e-8)


class SimpleLinearRegionAnalyzer:
    """
    精简的线性区域分析器 - 最小化内存占用和计算开销
    
    只计算4个核心指标：
    1. 区域数量
    2. 平均梯度范数
    3. 边界处平均梯度范数变化
    4. 边界处平均损失变化
    
    用法:
        analyzer = SimpleLinearRegionAnalyzer(model, input_shape=(784,), device='cuda')
        result = analyzer.analyze_direction(x, direction, label, max_distance=1.0)
        print(f"区域数: {result.num_regions}")
        print(f"平均梯度范数: {result.mean_gradient_norm}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu"
    ):
        """
        Args:
            model: PyTorch 模型（ReLU 激活）
            input_shape: 输入形状（不含 batch 维度）
            device: 计算设备
        """
        self.device = device
        self.input_shape = input_shape
        
        # 复制模型避免修改原模型
        self._model = deepcopy(model)
        self._model.eval()
        self._model.requires_grad_(False)
        self._model.to(device)
        
        # 获取模型 dtype
        self._model_dtype = next(self._model.parameters()).dtype
        
        # 查找 ReLU 层
        self._relu_modules = []
        self._find_relu_layers()
        
        # 预创建常量避免重复创建
        self._epsilon = torch.tensor([EPSILON], dtype=torch.float64, device=device)
        self._inf = torch.tensor([np.inf], dtype=torch.float64, device=device)
        
    def _find_relu_layers(self):
        """查找所有 ReLU 层"""
        relu_order = []
        
        def order_hook(module, inp, out):
            relu_order.append(module)
        
        temp_hooks = []
        for module in self._model.modules():
            if isinstance(module, nn.ReLU):
                temp_hooks.append(module.register_forward_hook(order_hook))
        
        # 运行一次前向传播确定 ReLU 顺序
        dummy = torch.ones((1,) + self.input_shape, device=self.device, dtype=self._model_dtype)
        with torch.no_grad():
            self._model(dummy)
        
        for h in temp_hooks:
            h.remove()
        
        self._relu_modules = relu_order
        self._num_relus = len(self._relu_modules)
    
    def _ensure_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """确保输入有 batch 维度"""
        if x.dim() == len(self.input_shape):
            return x.unsqueeze(0)
        return x
    
    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        """转移到设备并设置正确的 dtype"""
        return x.to(device=self.device, dtype=self._model_dtype)
    
    def _compute_gradient_norm(self, x: torch.Tensor) -> float:
        """
        计算梯度范数（Frobenius范数）
        
        优化：直接累积平方和，不构造完整雅可比矩阵
        """
        x_input = self._ensure_batch_dim(x)
        x_input = self._to_device(x_input)
        
        # 需要梯度
        x_grad = x_input.detach().clone().requires_grad_(True)
        
        logits = self._model(x_grad)
        num_classes = logits.shape[1]
        
        # 计算所有输出的梯度并累积平方和
        grad_norm_sq = 0.0
        for i in range(num_classes):
            if x_grad.grad is not None:
                x_grad.grad.zero_()
            
            logits[0, i].backward(retain_graph=(i < num_classes - 1))
            
            if x_grad.grad is not None:
                grad_norm_sq += (x_grad.grad ** 2).sum().item()
        
        return grad_norm_sq ** 0.5
    
    def _compute_gradient_norm_batch(self, points: torch.Tensor) -> torch.Tensor:
        """
        批量计算梯度范数
        
        Args:
            points: (batch, *input_shape)
            
        Returns:
            norms: (batch,)
        """
        batch_size = points.shape[0]
        points = self._to_device(points)
        
        norms = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            norms[i] = self._compute_gradient_norm(points[i:i+1])
        
        return norms
    
    def _compute_loss(self, x: torch.Tensor, label: int) -> float:
        """计算单点的损失"""
        x_input = self._ensure_batch_dim(x)
        x_input = self._to_device(x_input)
        
        with torch.no_grad():
            logits = self._model(x_input)
            loss = F.cross_entropy(logits, torch.tensor([label], device=self.device))
        
        return loss.item()
    
    def _compute_loss_batch(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """批量计算损失"""
        points = self._to_device(points)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            logits = self._model(points)
            losses = F.cross_entropy(logits, labels, reduction='none')
        
        return losses
    
    def _get_activation_pattern(self, x: torch.Tensor) -> List[np.ndarray]:
        """
        获取激活模式（简化版）
        只存储布尔值，不存储完整tensor
        """
        patterns = []
        
        def hook(module, input, output):
            # 只记录是否激活，转为 numpy 节省内存
            patterns.append((input[0] > 0).cpu().numpy())
        
        hooks = []
        for module in self._relu_modules:
            hooks.append(module.register_forward_hook(hook))
        
        x_input = self._ensure_batch_dim(x)
        x_input = self._to_device(x_input)
        
        with torch.no_grad():
            self._model(x_input)
        
        for h in hooks:
            h.remove()
        
        return patterns
    
    def _pattern_same(self, p1: List[np.ndarray], p2: List[np.ndarray]) -> bool:
        """比较两个激活模式是否相同"""
        if len(p1) != len(p2):
            return False
        for a, b in zip(p1, p2):
            if not np.array_equal(a, b):
                return False
        return True
    
    def _compute_lambda_to_boundary(
        self, 
        x: torch.Tensor, 
        direction: torch.Tensor
    ) -> float:
        """
        计算到最近边界的距离
        
        使用数值方法：通过检测激活模式变化来确定边界
        """
        x = self._ensure_batch_dim(x)
        direction = self._ensure_batch_dim(direction)
        x = self._to_device(x)
        direction = self._to_device(direction)
        
        # 获取当前激活模式
        current_pattern = self._get_activation_pattern(x)
        
        # 二分搜索找到最近的边界
        low = 0.0
        high = 1.0
        
        # 首先找到一个会改变激活模式的距离
        test_distances = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        boundary_found = False
        
        for dist in test_distances:
            x_test = x + dist * direction
            test_pattern = self._get_activation_pattern(x_test)
            if not self._pattern_same(current_pattern, test_pattern):
                high = dist
                boundary_found = True
                break
        
        if not boundary_found:
            return float('inf')
        
        # 二分搜索精确边界
        for _ in range(20):  # 20次迭代足够精确
            mid = (low + high) / 2
            x_mid = x + mid * direction
            mid_pattern = self._get_activation_pattern(x_mid)
            
            if self._pattern_same(current_pattern, mid_pattern):
                low = mid
            else:
                high = mid
            
            if high - low < EPSILON:
                break
        
        return high
    
    def analyze_direction(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        label: Optional[int] = None,
        max_distance: float = 1.0,
        max_regions: int = 100
    ) -> SimpleAnalysisResult:
        """
        分析沿给定方向的线性区域
        
        核心流程：
        1. 遍历线性区域（只记录边界位置）
        2. 在区域中点计算梯度范数
        3. 在边界前后计算梯度范数变化
        4. 在边界前后计算损失变化（如果提供了label）
        5. 返回统计结果
        
        Args:
            x: 起始点 (batch=1 or input_shape)
            direction: 方向向量（应该已归一化）
            label: 标签（用于计算损失）
            max_distance: 最大遍历距离
            max_regions: 最大区域数
            
        Returns:
            SimpleAnalysisResult
        """
        x = self._ensure_batch_dim(x)
        direction = self._ensure_batch_dim(direction)
        x = self._to_device(x)
        direction = self._to_device(direction)
        
        # 归一化方向
        direction = normalize_direction(direction)
        
        # 初始化统计
        num_regions = 0
        gradient_norms = []
        gradient_changes = []
        loss_changes = []
        
        current_t = 0.0
        
        # 获取初始激活模式
        prev_pattern = self._get_activation_pattern(x)
        
        while current_t < max_distance and num_regions < max_regions:
            # 计算当前位置
            current_x = x + current_t * direction
            
            # 计算到下一个边界的距离
            lambda_val = self._compute_lambda_to_boundary(current_x, direction)
            
            if lambda_val <= 0 or lambda_val == float('inf'):
                # 最后一个区域或无法找到边界
                num_regions += 1
                
                # 计算该区域中点的梯度范数
                remaining_dist = max_distance - current_t
                mid_t = current_t + remaining_dist / 2
                mid_x = x + mid_t * direction
                grad_norm = self._compute_gradient_norm(mid_x)
                gradient_norms.append(grad_norm)
                break
            
            # 调整lambda确保不超过max_distance
            actual_lambda = min(lambda_val, max_distance - current_t)
            
            # 记录区域
            num_regions += 1
            
            # 区域中点的梯度范数
            mid_t = current_t + actual_lambda / 2
            mid_x = x + mid_t * direction
            grad_norm = self._compute_gradient_norm(mid_x)
            gradient_norms.append(grad_norm)
            
            # 如果实际lambda等于剩余距离，说明到达终点
            if actual_lambda >= max_distance - current_t - EPSILON:
                break
            
            # 边界前后的梯度和损失
            boundary_t = current_t + actual_lambda
            eps = max(1e-5, actual_lambda * 0.1)
            
            x_before = x + (boundary_t - eps) * direction
            x_after = x + (boundary_t + eps) * direction
            
            grad_before = self._compute_gradient_norm(x_before)
            grad_after = self._compute_gradient_norm(x_after)
            gradient_changes.append(abs(grad_after - grad_before))
            
            if label is not None:
                loss_before = self._compute_loss(x_before, label)
                loss_after = self._compute_loss(x_after, label)
                loss_changes.append(loss_after - loss_before)
            
            # 移动到下一个区域
            current_t = boundary_t + eps
            
            # 检查激活模式是否改变
            new_x = x + current_t * direction
            new_pattern = self._get_activation_pattern(new_x)
            if self._pattern_same(prev_pattern, new_pattern):
                # 激活模式没变，可能是数值问题，继续尝试
                pass
            prev_pattern = new_pattern
        
        return SimpleAnalysisResult(
            num_regions=num_regions,
            mean_gradient_norm=sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0,
            mean_gradient_norm_change=sum(gradient_changes) / len(gradient_changes) if gradient_changes else 0.0,
            mean_loss_change=sum(loss_changes) / len(loss_changes) if loss_changes else 0.0
        )
    
    def analyze_batch(
        self,
        x_batch: torch.Tensor,
        directions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        max_distance: float = 1.0,
        max_regions: int = 100
    ) -> List[SimpleAnalysisResult]:
        """
        批量分析 - 收集所有计算点后一次性批量计算
        
        核心思路：
        1. 串行遍历所有样本，收集需要计算梯度的点
        2. 将所有点合并为一个大batch
        3. 一次性计算所有点的梯度范数
        4. 分配结果回各个样本
        
        Args:
            x_batch: 输入批次 (batch, *input_shape)
            directions: 方向批次 (batch, *input_shape)
            labels: 标签批次 (batch,)
            max_distance: 最大遍历距离
            max_regions: 最大区域数
            
        Returns:
            List[SimpleAnalysisResult]
        """
        batch_size = x_batch.shape[0]
        x_batch = self._to_device(x_batch)
        directions = self._to_device(directions)
        directions = normalize_direction(directions)
        
        # 第1步：遍历并收集所有计算点
        all_mid_points = []
        all_boundary_before = []
        all_boundary_after = []
        
        point_mapping = []  # [(sample_idx, num_mid_points, num_boundaries), ...]
        region_counts = []
        
        for i in range(batch_size):
            x = x_batch[i:i+1]
            direction = directions[i:i+1]
            label = labels[i].item() if labels is not None else None
            
            # 遍历该样本收集点
            points_info = self._traverse_and_collect_points(
                x, direction, label, max_distance, max_regions
            )
            
            all_mid_points.extend(points_info['mid_points'])
            all_boundary_before.extend(points_info['boundary_before'])
            all_boundary_after.extend(points_info['boundary_after'])
            
            point_mapping.append({
                'num_mid': len(points_info['mid_points']),
                'num_boundary': len(points_info['boundary_before'])
            })
            region_counts.append(points_info['num_regions'])
        
        # 第2步：批量计算所有梯度范数
        mid_grad_norms = []
        if len(all_mid_points) > 0:
            all_mid_tensor = torch.cat(all_mid_points, dim=0)
            mid_grad_norms = self._compute_gradient_norm_batch(all_mid_tensor)
        
        boundary_grad_before = []
        boundary_grad_after = []
        if len(all_boundary_before) > 0:
            before_tensor = torch.cat(all_boundary_before, dim=0)
            after_tensor = torch.cat(all_boundary_after, dim=0)
            boundary_grad_before = self._compute_gradient_norm_batch(before_tensor)
            boundary_grad_after = self._compute_gradient_norm_batch(after_tensor)
        
        # 第3步：批量计算损失（如果需要）
        boundary_loss_before = []
        boundary_loss_after = []
        if labels is not None and len(all_boundary_before) > 0:
            # 需要为每个边界点准备对应的label
            boundary_labels = []
            for i, mapping in enumerate(point_mapping):
                boundary_labels.extend([labels[i].item()] * mapping['num_boundary'])
            
            if len(boundary_labels) > 0:
                boundary_labels_tensor = torch.tensor(boundary_labels, dtype=torch.long, device=self.device)
                boundary_loss_before = self._compute_loss_batch(before_tensor, boundary_labels_tensor)
                boundary_loss_after = self._compute_loss_batch(after_tensor, boundary_labels_tensor)
        
        # 第4步：分配结果回各个样本
        results = []
        mid_idx = 0
        boundary_idx = 0
        
        for i, mapping in enumerate(point_mapping):
            num_mid = mapping['num_mid']
            num_boundary = mapping['num_boundary']
            
            # 提取该样本的梯度范数
            if num_mid > 0:
                sample_grad_norms = mid_grad_norms[mid_idx:mid_idx + num_mid].tolist()
                mean_grad_norm = sum(sample_grad_norms) / len(sample_grad_norms)
            else:
                mean_grad_norm = 0.0
            
            # 提取该样本的梯度变化
            if num_boundary > 0:
                before = boundary_grad_before[boundary_idx:boundary_idx + num_boundary]
                after = boundary_grad_after[boundary_idx:boundary_idx + num_boundary]
                grad_changes = torch.abs(after - before).tolist()
                mean_grad_change = sum(grad_changes) / len(grad_changes)
            else:
                mean_grad_change = 0.0
            
            # 提取该样本的损失变化
            mean_loss_change = 0.0
            if labels is not None and num_boundary > 0 and len(boundary_loss_before) > 0:
                loss_before = boundary_loss_before[boundary_idx:boundary_idx + num_boundary]
                loss_after = boundary_loss_after[boundary_idx:boundary_idx + num_boundary]
                loss_changes = (loss_after - loss_before).tolist()
                mean_loss_change = sum(loss_changes) / len(loss_changes)
            
            results.append(SimpleAnalysisResult(
                num_regions=region_counts[i],
                mean_gradient_norm=mean_grad_norm,
                mean_gradient_norm_change=mean_grad_change,
                mean_loss_change=mean_loss_change
            ))
            
            mid_idx += num_mid
            boundary_idx += num_boundary
        
        return results
    
    def _traverse_and_collect_points(
        self,
        x: torch.Tensor,
        direction: torch.Tensor,
        label: Optional[int],
        max_distance: float,
        max_regions: int
    ) -> dict:
        """
        遍历线性区域并收集需要计算的点
        
        Returns:
            dict with keys:
                - num_regions: int
                - mid_points: List[Tensor] - 每个区域的中点
                - boundary_before: List[Tensor] - 边界前的点
                - boundary_after: List[Tensor] - 边界后的点
        """
        mid_points = []
        boundary_before = []
        boundary_after = []
        
        num_regions = 0
        current_t = 0.0
        
        prev_pattern = self._get_activation_pattern(x)
        
        while current_t < max_distance and num_regions < max_regions:
            current_x = x + current_t * direction
            
            lambda_val = self._compute_lambda_to_boundary(current_x, direction)
            
            if lambda_val <= 0 or lambda_val == float('inf'):
                num_regions += 1
                remaining_dist = max_distance - current_t
                mid_t = current_t + remaining_dist / 2
                mid_x = x + mid_t * direction
                mid_points.append(mid_x.clone())
                break
            
            actual_lambda = min(lambda_val, max_distance - current_t)
            num_regions += 1
            
            mid_t = current_t + actual_lambda / 2
            mid_x = x + mid_t * direction
            mid_points.append(mid_x.clone())
            
            if actual_lambda >= max_distance - current_t - EPSILON:
                break
            
            boundary_t = current_t + actual_lambda
            eps = max(1e-5, actual_lambda * 0.1)
            
            x_before = x + (boundary_t - eps) * direction
            x_after = x + (boundary_t + eps) * direction
            
            boundary_before.append(x_before.clone())
            boundary_after.append(x_after.clone())
            
            current_t = boundary_t + eps
            
            new_x = x + current_t * direction
            new_pattern = self._get_activation_pattern(new_x)
            prev_pattern = new_pattern
        
        return {
            'num_regions': num_regions,
            'mid_points': mid_points,
            'boundary_before': boundary_before,
            'boundary_after': boundary_after
        }
    
    def find_decision_boundary_direction(
        self,
        x: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        计算朝向决策边界的方向
        
        方向定义：∇_x (logit[top2] - logit[top1])
        沿此方向移动会使 top1 和 top2 的差距减小
        
        Args:
            x: 输入数据点 (batch, *input_shape)
            normalize: 是否归一化方向向量
            
        Returns:
            direction: 方向向量
        """
        x = self._ensure_batch_dim(x)
        x = self._to_device(x)
        x_grad = x.detach().clone().requires_grad_(True)
        
        logits = self._model(x_grad)
        
        top2_values, _ = torch.topk(logits, k=2, dim=1)
        top1_logit = top2_values[:, 0]
        top2_logit = top2_values[:, 1]
        
        loss = (top2_logit - top1_logit).sum()
        loss.backward()
        
        direction = x_grad.grad.detach().clone()
        
        if normalize:
            direction = normalize_direction(direction)
        
        return direction
    
    def analyze(
        self,
        x: torch.Tensor,
        label: Optional[int] = None,
        max_distance: float = 1.0,
        max_regions: int = 100
    ) -> SimpleAnalysisResult:
        """
        便捷方法：自动计算决策边界方向并分析
        
        Args:
            x: 输入点
            label: 标签
            max_distance: 最大距离
            max_regions: 最大区域数
            
        Returns:
            SimpleAnalysisResult
        """
        x = self._ensure_batch_dim(x)
        x = self._to_device(x)
        
        direction = self.find_decision_boundary_direction(x)
        
        return self.analyze_direction(x, direction, label, max_distance, max_regions)
    
    def cleanup(self):
        """清理资源"""
        # 当前实现无需特别清理
        pass
