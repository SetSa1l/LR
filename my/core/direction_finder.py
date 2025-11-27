# -*- coding: utf-8 -*-
"""
DecisionBoundaryDirectionFinder: 计算朝向决策边界的方向

决策边界定义：最大 logit 和第二大 logit 相等的位置
方向：使 (logit[top2] - logit[top1]) 增大的梯度方向
"""

import torch
import torch.nn as nn
from typing import Tuple
from . model_wrapper import normalize_direction


class DecisionBoundaryDirectionFinder:
    """
    计算朝向决策边界的方向
    
    用法:
        finder = DecisionBoundaryDirectionFinder(model, device='cuda')
        directions = finder.find_direction(x)
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Args:
            model: PyTorch 模型
            device: 计算设备
        """
        self. model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def find_direction(
        self, 
        x: torch.Tensor, 
        normalize: bool = True
    ) -> torch.Tensor:
        """
        批量计算朝向决策边界的方向
        
        方向定义：∇_x (logit[top2] - logit[top1])
        沿此方向移动会使 top1 和 top2 的差距减小
        
        Args:
            x: 输入数据点, shape (batch, *input_shape)
            normalize: 是否归一化方向向量
        
        Returns:
            direction: 方向向量, shape 同 x
        """
        x = x.to(self.device)
        x_grad = x.detach(). clone(). requires_grad_(True)
        
        # 前向传播
        logits = self. model(x_grad)  # (batch, num_classes)
        
        # 找到每个样本的 top1 和 top2
        top2_values, top2_indices = torch. topk(logits, k=2, dim=1)
        top1_logit = top2_values[:, 0]  # (batch,)
        top2_logit = top2_values[:, 1]  # (batch,)
        
        # 目标：最大化 (top2 - top1)，即朝决策边界移动
        # 对 batch 求和以便一次反向传播
        loss = (top2_logit - top1_logit).sum()
        
        # 反向传播
        loss.backward()
        
        # 梯度就是方向
        direction = x_grad. grad.detach(). clone()
        
        # 归一化
        if normalize:
            direction = normalize_direction(direction)
        
        return direction
    
    def find_direction_with_info(
        self, 
        x: torch.Tensor, 
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量计算方向，同时返回 top1/top2 信息
        
        Args:
            x: 输入数据点
            normalize: 是否归一化
        
        Returns:
            directions: 方向向量
            top1_indices: 每个样本的 top1 类别
            top2_indices: 每个样本的 top2 类别
            margins: top1 - top2 的差值（到决策边界的 logit 距离）
        """
        x = x.to(self.device)
        x_grad = x.detach().clone().requires_grad_(True)
        
        logits = self.model(x_grad)
        
        top2_values, top2_indices = torch.topk(logits, k=2, dim=1)
        top1_indices = top2_indices[:, 0]
        top2_indices_out = top2_indices[:, 1]
        margins = top2_values[:, 0] - top2_values[:, 1]
        
        loss = (top2_values[:, 1] - top2_values[:, 0]).sum()
        loss.backward()
        
        direction = x_grad.grad. detach().clone()
        
        if normalize:
            direction = normalize_direction(direction)
        
        return direction, top1_indices, top2_indices_out, margins. detach()
    
    def find_direction_to_target_class(
        self, 
        x: torch.Tensor, 
        target_classes: torch.Tensor,
        normalize: bool = True
    ) -> torch. Tensor:
        """
        批量计算朝向指定目标类别决策边界的方向
        
        Args:
            x: 输入数据点 (batch, *input_shape)
            target_classes: 每个样本的目标类别 (batch,) 或单个 int
            normalize: 是否归一化
        
        Returns:
            direction: 方向向量
        """
        x = x.to(self.device)
        batch_size = x. shape[0]
        
        # 处理单个 target_class 的情况
        if isinstance(target_classes, int):
            target_classes = torch.full((batch_size,), target_classes, 
                                        dtype=torch.long, device=self.device)
        else:
            target_classes = target_classes.to(self. device)
        
        x_grad = x. detach().clone(). requires_grad_(True)
        
        logits = self. model(x_grad)
        
        # 当前预测类别
        current_classes = logits.argmax(dim=1)
        
        # 获取当前类别和目标类别的 logit
        batch_indices = torch. arange(batch_size, device=self.device)
        current_logits = logits[batch_indices, current_classes]
        target_logits = logits[batch_indices, target_classes]
        
        # 目标：最大化 (target_logit - current_logit)
        loss = (target_logits - current_logits).sum()
        loss.backward()
        
        direction = x_grad.grad. detach().clone()
        
        if normalize:
            direction = normalize_direction(direction)
        
        return direction
    
    def find_adversarial_direction(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True
    ) -> torch. Tensor:
        """
        计算对抗方向：使正确类别 logit 下降最快的方向
        
        Args:
            x: 输入数据点 (batch, *input_shape)
            labels: 正确标签 (batch,)
            normalize: 是否归一化
        
        Returns:
            direction: 对抗方向
        """
        x = x.to(self. device)
        labels = labels.to(self.device)
        x_grad = x. detach().clone(). requires_grad_(True)
        
        logits = self. model(x_grad)
        
        batch_indices = torch. arange(x. shape[0], device=self.device)
        correct_logits = logits[batch_indices, labels]
        
        # 目标：最小化正确类别的 logit
        loss = -correct_logits. sum()
        loss.backward()
        
        direction = x_grad.grad. detach().clone()
        
        if normalize:
            direction = normalize_direction(direction)
        
        return direction


def find_decision_boundary_direction(
    model: nn.Module, 
    x: torch.Tensor, 
    device: str = "cpu"
) -> torch. Tensor:
    """
    便捷函数：计算朝向决策边界的方向
    
    Args:
        model: PyTorch 模型
        x: 输入数据点
        device: 计算设备
    
    Returns:
        direction: 归一化的方向向量
    """
    finder = DecisionBoundaryDirectionFinder(model, device)
    return finder. find_direction(x)