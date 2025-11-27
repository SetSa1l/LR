# -*- coding: utf-8 -*-
"""
RegionPropertyAnalyzer: 计算线性区域的性质（高效批量版本）

主要指标：
1. 雅可比矩阵范数：衡量区域内的局部 Lipschitz 常数
2. 相邻区域梯度范数差：衡量跨边界的梯度变化
3. 相邻区域平均 Loss 差：衡量跨边界的 loss 变化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

from . region_traverser import TraversalResult, BatchTraversalResult


@dataclass
class RegionProperties:
    """单个区域的性质"""
    region_id: int
    entry_t: float
    exit_t: float
    jacobian_frobenius_norm: float = 0.0
    jacobian_spectral_norm: float = 0.0
    mean_loss: float = 0.0
    logit_variation: float = 0.0


@dataclass
class AdjacentRegionDiff:
    """相邻区域之间的差异"""
    region_id_before: int
    region_id_after: int
    boundary_t: float
    jacobian_norm_diff: float = 0.0
    loss_diff: float = 0.0
    logit_diff: float = 0.0


@dataclass
class TraversalProperties:
    """完整遍历的性质汇总"""
    num_regions: int
    total_distance: float
    region_properties: List[RegionProperties] = field(default_factory=list)
    adjacent_diffs: List[AdjacentRegionDiff] = field(default_factory=list)
    mean_jacobian_norm: float = 0.0
    max_jacobian_norm: float = 0.0
    mean_jacobian_diff: float = 0.0
    max_jacobian_diff: float = 0.0
    mean_loss_diff: float = 0.0
    total_loss_change: float = 0.0


class RegionPropertyAnalyzer:
    """
    区域性质分析器（高效批量版本）
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: str = "cpu",
        num_samples_per_region: int = 3
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.num_samples = num_samples_per_region
        
        # 获取模型的 dtype
        self._model_dtype = next(model.parameters()).dtype
        
        # 缓存输出维度
        self._output_dim = None
    
    def _get_output_dim(self, input_shape: Tuple) -> int:
        """获取模型输出维度"""
        if self._output_dim is None:
            dummy = torch.zeros((1,) + input_shape, device=self.device, dtype=self._model_dtype)
            with torch.no_grad():
                out = self.model(dummy)
            self._output_dim = out.shape[1]
        return self._output_dim
    
    def _to_model_dtype(self, x: torch.Tensor) -> torch. Tensor:
        """转换到模型的 dtype"""
        return x.to(device=self.device, dtype=self._model_dtype)
    
    def compute_jacobian_batch(
        self, 
        x: torch. Tensor
    ) -> torch. Tensor:
        """
        批量计算雅可比矩阵
        
        Args:
            x: (batch, *input_shape)
        
        Returns:
            jacobian: (batch, output_dim, input_dim)
        """
        batch_size = x. shape[0]
        input_shape = x. shape[1:]
        input_dim = x[0].numel()
        output_dim = self._get_output_dim(input_shape)
        
        # 确保 dtype 正确
        x = self._to_model_dtype(x)
        x_flat = x.view(batch_size, -1)
        
        # 创建需要梯度的输入
        x_grad = x_flat.clone(). requires_grad_(True)
        x_input = x_grad. view(batch_size, *input_shape)
        
        # 前向传播
        logits = self. model(x_input)  # (batch, output_dim)
        
        # 批量计算每个输出维度的梯度
        jacobian = torch.zeros(batch_size, output_dim, input_dim, 
                               device=self.device, dtype=self._model_dtype)
        
        for i in range(output_dim):
            # 对第 i 个输出维度求梯度
            grad_outputs = torch.zeros_like(logits)
            grad_outputs[:, i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=logits,
                inputs=x_grad,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]  # (batch, input_dim)
            
            jacobian[:, i, :] = grads
        
        return jacobian
    
    def compute_jacobian_norms_batch(
        self, 
        x: torch. Tensor
    ) -> Tuple[torch. Tensor, torch. Tensor]:
        """
        批量计算雅可比矩阵的范数
        
        Args:
            x: (batch, *input_shape)
        
        Returns:
            frobenius_norms: (batch,)
            spectral_norms: (batch,)
        """
        jacobian = self.compute_jacobian_batch(x)  # (batch, output_dim, input_dim)
        batch_size = jacobian.shape[0]
        
        # Frobenius 范数
        frobenius_norms = torch.norm(jacobian. view(batch_size, -1), p=2, dim=1)
        
        # 谱范数（最大奇异值）
        try:
            singular_values = torch. linalg.svdvals(jacobian)  # (batch, min(out, in))
            spectral_norms = singular_values[:, 0]
        except:
            spectral_norms = frobenius_norms. clone()
        
        return frobenius_norms, spectral_norms
    
    def compute_loss_batch(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        批量计算 cross-entropy loss
        
        Args:
            x: (batch, *input_shape)
            labels: (batch,)
        
        Returns:
            losses: (batch,)
        """
        x = self._to_model_dtype(x)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            losses = F.cross_entropy(logits, labels, reduction='none')
        
        return losses
    
    def compute_logit_diff_batch(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch. Tensor:
        """
        批量计算两组点之间的 logit 差异
        
        Args:
            x1, x2: (batch, *input_shape)
        
        Returns:
            diffs: (batch,) L2 范数差
        """
        x1 = self._to_model_dtype(x1)
        x2 = self._to_model_dtype(x2)
        
        with torch.no_grad():
            logits1 = self. model(x1)
            logits2 = self.model(x2)
            diffs = torch.norm(logits2 - logits1, dim=1)
        return diffs
    
    def _generate_sample_points(
        self,
        start: torch.Tensor,
        direction: torch.Tensor,
        entry_ts: torch.Tensor,
        exit_ts: torch. Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        批量生成区域内的采样点
        
        Args:
            start: (batch, *input_shape)
            direction: (batch, *input_shape)
            entry_ts: (batch,)
            exit_ts: (batch,)
            num_samples: 每个区域采样数
        
        Returns:
            points: (batch * num_samples, *input_shape) 或 (batch, *input_shape) if num_samples=1
        """
        batch_size = start.shape[0]
        input_shape = start.shape[1:]
        
        # 确保 start 和 direction 是正确的 dtype
        start = self._to_model_dtype(start)
        direction = self._to_model_dtype(direction)
        
        # 将 t 转换为与 start 相同的 dtype
        entry_ts = entry_ts.to(dtype=self._model_dtype, device=self.device)
        exit_ts = exit_ts.to(dtype=self._model_dtype, device=self. device)
        
        if num_samples == 1:
            # 只返回中点
            mid_t = (entry_ts + exit_ts) / 2  # (batch,)
            mid_t = mid_t. view(batch_size, *([1] * len(input_shape)))
            return start + mid_t * direction
        
        # 多个采样点
        points = []
        for i in range(num_samples):
            ratio = (i + 0.5) / num_samples
            t = entry_ts + ratio * (exit_ts - entry_ts)
            t = t.view(batch_size, *([1] * len(input_shape)))
            points.append(start + t * direction)
        
        return torch.cat(points, dim=0)  # (batch * num_samples, *input_shape)
    
    def analyze_traversal(
        self,
        traversal: TraversalResult,
        start: torch.Tensor,
        direction: torch.Tensor,
        label: Optional[Union[torch.Tensor, int]] = None
    ) -> TraversalProperties:
        """
        分析完整的遍历结果（高效批量版本）
        """
        if traversal.num_regions == 0:
            return TraversalProperties(
                num_regions=0,
                total_distance=traversal.total_distance
            )
        
        # 转换为正确的 dtype
        start = self._to_model_dtype(start)
        direction = self._to_model_dtype(direction)
        
        if start.dim() > 1 and start.shape[0] == 1:
            start = start. squeeze(0)
        if direction.dim() > 1 and direction.shape[0] == 1:
            direction = direction.squeeze(0)
        
        num_regions = traversal.num_regions
        input_shape = start.shape
        
        # 收集所有区域的 entry_t 和 exit_t（保持 float32）
        entry_ts = torch.tensor([r.entry_t for r in traversal. regions], 
                                dtype=self._model_dtype, device=self.device)
        exit_ts = torch.tensor([r.exit_t for r in traversal.regions], 
                               dtype=self._model_dtype, device=self.device)
        
        # 扩展 start 和 direction 到 batch
        starts = start.unsqueeze(0).expand(num_regions, *input_shape). clone()
        directions = direction.unsqueeze(0).expand(num_regions, *input_shape). clone()
        
        # 1. 计算每个区域中点的雅可比范数
        mid_points = self._generate_sample_points(starts, directions, entry_ts, exit_ts, num_samples=1)
        fro_norms, spec_norms = self.compute_jacobian_norms_batch(mid_points)
        
        # 2. 计算每个区域的平均 loss（如果有 label）
        mean_losses = torch.zeros(num_regions, device=self.device)
        if label is not None:
            if isinstance(label, int):
                labels = torch.full((num_regions,), label, dtype=torch.long, device=self.device)
            else:
                labels = label. expand(num_regions). to(self.device)
            
            if self.num_samples == 1:
                mean_losses = self. compute_loss_batch(mid_points, labels)
            else:
                # 多点采样取平均
                sample_points = self._generate_sample_points(
                    starts, directions, entry_ts, exit_ts, num_samples=self. num_samples
                )
                sample_labels = labels.repeat(self.num_samples)
                all_losses = self.compute_loss_batch(sample_points, sample_labels)
                mean_losses = all_losses. view(self.num_samples, num_regions). mean(dim=0)
        
        # 3. 计算每个区域的 logit 变化
        eps_ratio = 0.05
        region_lengths = exit_ts - entry_ts
        entry_offset = entry_ts + eps_ratio * region_lengths
        exit_offset = exit_ts - eps_ratio * region_lengths
        
        entry_points = self._generate_sample_points(
            starts, directions, entry_offset, entry_offset, num_samples=1
        )
        exit_points = self._generate_sample_points(
            starts, directions, exit_offset, exit_offset, num_samples=1
        )
        logit_variations = self.compute_logit_diff_batch(entry_points, exit_points)
        
        # 4. 计算相邻区域的差异
        jacobian_diffs = torch.tensor([], device=self.device)
        loss_diffs = torch.tensor([], device=self.device)
        logit_diffs = torch.tensor([], device=self.device)
        
        if num_regions > 1:
            # 边界点（边界两侧各取一点）
            boundary_ts = exit_ts[:-1]  # (num_regions - 1,)
            eps = torch.clamp(region_lengths[:-1] * 0.1, min=1e-5)
            
            before_ts = boundary_ts - eps
            after_ts = boundary_ts + eps
            
            n_boundaries = num_regions - 1
            boundary_starts = start. unsqueeze(0).expand(n_boundaries, *input_shape). clone()
            boundary_dirs = direction.unsqueeze(0).expand(n_boundaries, *input_shape). clone()
            
            before_points = self._generate_sample_points(
                boundary_starts, boundary_dirs, before_ts, before_ts, num_samples=1
            )
            after_points = self._generate_sample_points(
                boundary_starts, boundary_dirs, after_ts, after_ts, num_samples=1
            )
            
            # 雅可比范数差
            fro_before, _ = self.compute_jacobian_norms_batch(before_points)
            fro_after, _ = self.compute_jacobian_norms_batch(after_points)
            jacobian_diffs = torch.abs(fro_after - fro_before)
            
            # Loss 差
            if label is not None:
                if isinstance(label, int):
                    boundary_labels = torch. full((n_boundaries,), label, dtype=torch.long, device=self.device)
                else:
                    boundary_labels = label.expand(n_boundaries). to(self.device)
                
                loss_before = self.compute_loss_batch(before_points, boundary_labels)
                loss_after = self. compute_loss_batch(after_points, boundary_labels)
                loss_diffs = loss_after - loss_before
            else:
                loss_diffs = torch.zeros(n_boundaries, device=self.device)
            
            # Logit 差
            logit_diffs = self.compute_logit_diff_batch(before_points, after_points)
        
        # 构建结果
        region_props = []
        for i in range(num_regions):
            region_props.append(RegionProperties(
                region_id=i,
                entry_t=entry_ts[i]. item(),
                exit_t=exit_ts[i].item(),
                jacobian_frobenius_norm=fro_norms[i].item(),
                jacobian_spectral_norm=spec_norms[i].item(),
                mean_loss=mean_losses[i].item(),
                logit_variation=logit_variations[i].item()
            ))
        
        adj_diffs = []
        for i in range(num_regions - 1):
            adj_diffs. append(AdjacentRegionDiff(
                region_id_before=i,
                region_id_after=i + 1,
                boundary_t=exit_ts[i]. item(),
                jacobian_norm_diff=jacobian_diffs[i].item() if len(jacobian_diffs) > 0 else 0.0,
                loss_diff=loss_diffs[i]. item() if len(loss_diffs) > 0 else 0.0,
                logit_diff=logit_diffs[i]. item() if len(logit_diffs) > 0 else 0.0
            ))
        
        # 汇总统计
        fro_list = fro_norms. tolist()
        jac_diff_list = jacobian_diffs.tolist() if len(jacobian_diffs) > 0 else []
        loss_diff_list = loss_diffs.tolist() if len(loss_diffs) > 0 else []
        
        return TraversalProperties(
            num_regions=num_regions,
            total_distance=traversal.total_distance,
            region_properties=region_props,
            adjacent_diffs=adj_diffs,
            mean_jacobian_norm=sum(fro_list) / len(fro_list) if fro_list else 0.0,
            max_jacobian_norm=max(fro_list) if fro_list else 0.0,
            mean_jacobian_diff=sum(jac_diff_list) / len(jac_diff_list) if jac_diff_list else 0.0,
            max_jacobian_diff=max(jac_diff_list) if jac_diff_list else 0.0,
            mean_loss_diff=sum(abs(d) for d in loss_diff_list) / len(loss_diff_list) if loss_diff_list else 0.0,
            total_loss_change=sum(loss_diff_list) if loss_diff_list else 0.0
        )
    
    def analyze_batch(
        self,
        batch_traversal: BatchTraversalResult,
        starts: torch.Tensor,
        directions: torch.Tensor,
        labels: Optional[torch. Tensor] = None
    ) -> List[TraversalProperties]:
        """
        批量分析遍历结果
        """
        results = []
        
        for i in range(batch_traversal.batch_size):
            traversal = batch_traversal.get_single_result(i, starts, directions)
            label = labels[i]. item() if labels is not None else None
            
            props = self.analyze_traversal(
                traversal,
                starts[i:i+1],
                directions[i:i+1],
                label
            )
            results.append(props)
        
        return results