# -*- coding: utf-8 -*-
"""
LinearRegionTraverser: 沿给定方向遍历线性区域
真正的批量并行处理
"""

import torch
import numpy as np
from typing import List
from dataclasses import dataclass, field

from .model_wrapper import ModelWrapper, ActivationPattern, normalize_direction, EPSILON, CROSSING_EPSILON


@dataclass
class LinearRegionInfo:
    """单个线性区域的信息"""
    region_id: int
    activation_pattern: ActivationPattern
    entry_t: float
    exit_t: float
    
    @property
    def length(self) -> float:
        return self.exit_t - self.entry_t


@dataclass
class TraversalResult:
    """单个样本的遍历结果"""
    start_point: torch.Tensor
    direction: torch. Tensor
    regions: List[LinearRegionInfo] = field(default_factory=list)
    total_distance: float = 0.0
    num_regions: int = 0


@dataclass
class BatchTraversalResult:
    """批量遍历结果"""
    batch_size: int
    num_regions: torch.Tensor  # (batch,)
    entry_ts: torch.Tensor  # (batch, max_regions)
    exit_ts: torch.Tensor  # (batch, max_regions)
    total_distances: torch.Tensor  # (batch,)
    activation_patterns: List[torch.Tensor]  # 每层一个 (batch, max_regions, ...)
    
    def get_single_result(self, idx: int, start_point: torch.Tensor, direction: torch. Tensor) -> TraversalResult:
        """提取单个样本的结果"""
        n_regions = self. num_regions[idx]. item()
        regions = []
        
        for r in range(n_regions):
            patterns = [p[idx, r] for p in self. activation_patterns]
            regions.append(LinearRegionInfo(
                region_id=r,
                activation_pattern=ActivationPattern(patterns=[p. unsqueeze(0) for p in patterns]),
                entry_t=self.entry_ts[idx, r].item(),
                exit_t=self.exit_ts[idx, r].item()
            ))
        
        return TraversalResult(
            start_point=start_point[idx:idx+1] if start_point.shape[0] > idx else start_point,
            direction=direction[idx:idx+1] if direction.shape[0] > idx else direction,
            regions=regions,
            total_distance=self.total_distances[idx]. item(),
            num_regions=n_regions
        )


class LinearRegionTraverser:
    """
    线性区域遍历器（真正的批量并行处理）
    """
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.wrapper = model_wrapper
        self.device = model_wrapper.device
        self._model_dtype = model_wrapper._model_dtype
    
    def traverse(
        self,
        start: torch.Tensor,
        direction: torch.Tensor,
        max_distance: float = 10.0,
        max_regions: int = 100,
        normalize_dir: bool = True
    ) -> TraversalResult:
        """单样本遍历"""
        if start.dim() == len(self.wrapper.input_shape):
            start = start.unsqueeze(0)
            direction = direction.unsqueeze(0)
        
        batch_result = self. traverse_batch(
            starts=start,
            directions=direction,
            max_distance=max_distance,
            max_regions=max_regions,
            normalize_dir=normalize_dir
        )
        
        return batch_result. get_single_result(0, start, direction)
    
    def _check_pattern_same(
        self, 
        curr_patterns: List[torch.Tensor],
        prev_patterns: List[torch. Tensor],
        active_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        检查当前激活模式是否与上一步相同
        
        Args:
            curr_patterns: 当前激活模式列表，每个 shape (n_active, ...)
            prev_patterns: 上一步激活模式列表，每个 shape (batch_size, ...)
            active_indices: 活跃样本的全局索引 (n_active,)
        
        Returns:
            same_mask: (n_active,) bool tensor，True 表示模式相同
        """
        n_active = active_indices.shape[0]
        same_mask = torch. ones(n_active, dtype=torch. bool, device=self.device)
        
        for curr_pattern, prev_pattern in zip(curr_patterns, prev_patterns):
            # curr_pattern: (n_active, ...)
            # prev_pattern: (batch_size, ...)
            # 只比较活跃样本
            prev_selected = prev_pattern[active_indices]  # (n_active, ...)
            
            curr_flat = curr_pattern. reshape(n_active, -1)
            prev_flat = prev_selected.reshape(n_active, -1)
            
            layer_same = (curr_flat == prev_flat).all(dim=1)
            same_mask = same_mask & layer_same
        
        return same_mask
    
    def traverse_batch(
        self,
        starts: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float = 10.0,
        max_regions: int = 100,
        normalize_dir: bool = True
    ) -> BatchTraversalResult:
        """
        批量并行遍历
        """
        if starts.dim() == len(self.wrapper.input_shape):
            starts = starts.unsqueeze(0)
            directions = directions.unsqueeze(0)
        
        if normalize_dir:
            directions = normalize_direction(directions)
        
        batch_size = starts.shape[0]
        starts = starts.to(device=self.device, dtype=self._model_dtype)
        directions = directions.to(device=self.device, dtype=self._model_dtype)
        
        # 预分配结果缓冲区
        entry_ts = torch. full((batch_size, max_regions), float('inf'),
                              dtype=torch.float64, device=self. device)
        exit_ts = torch. full((batch_size, max_regions), float('inf'),
                             dtype=torch.float64, device=self.device)
        
        # 获取激活模式形状
        init_state = self.wrapper. get_region_state(starts[:1], directions[:1])
        pattern_shapes = [p.shape[1:] for p in init_state.activation_pattern. patterns]
        
        activation_patterns = [
            torch.zeros((batch_size, max_regions) + shape, dtype=torch.bool, device=self. device)
            for shape in pattern_shapes
        ]
        
        # 状态变量
        current_x = starts. clone()
        current_t = torch.zeros(batch_size, dtype=torch.float64, device=self. device)
        region_count = torch.zeros(batch_size, dtype=torch.long, device=self. device)
        
        # 活跃样本掩码
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self. device)
        
        # 上一步的激活模式
        prev_pattern_tensors = None
        
        iteration = 0
        while active_mask.any() and iteration < max_regions:
            iteration += 1
            
            # 获取活跃样本索引
            active_indices = torch.where(active_mask)[0]
            n_active = active_indices.shape[0]
            
            if n_active == 0:
                break
            
            # 提取活跃样本
            active_x = current_x[active_indices]
            active_dir = directions[active_indices]
            
            # 批量获取区域状态
            state = self.wrapper.get_region_state(active_x, active_dir)
            
            # 检测是否需要重试（激活模式未变）
            if prev_pattern_tensors is not None:
                same_mask = self._check_pattern_same(
                    state.activation_pattern. patterns,
                    prev_pattern_tensors,
                    active_indices
                )
                
                if same_mask.any():
                    # 对模式相同的样本增大步长
                    retry_local_indices = torch.where(same_mask)[0]
                    retry_global_indices = active_indices[retry_local_indices]
                    current_x[retry_global_indices] += CROSSING_EPSILON * 10 * directions[retry_global_indices]
                    current_t[retry_global_indices] += CROSSING_EPSILON * 10
                    
                    if same_mask.all():
                        continue
                    
                    # 过滤掉需要重试的样本
                    diff_mask = ~same_mask
                    keep_local = torch.where(diff_mask)[0]
                    active_indices = active_indices[keep_local]
                    n_active = active_indices.shape[0]
                    
                    # 重新获取状态
                    active_x = current_x[active_indices]
                    active_dir = directions[active_indices]
                    state = self.wrapper. get_region_state(active_x, active_dir)
            
            if n_active == 0:
                continue
            
            active_t = current_t[active_indices]
            active_region_count = region_count[active_indices]
            lambda_vals = state.lambda_to_boundary
            
            # 检查是否超出最大区域数
            valid_mask = active_region_count < max_regions
            if not valid_mask. all():
                exceeded_local = torch.where(~valid_mask)[0]
                exceeded_global = active_indices[exceeded_local]
                active_mask[exceeded_global] = False
                
                if not valid_mask.any():
                    continue
                
                keep_local = torch. where(valid_mask)[0]
                active_indices = active_indices[keep_local]
                active_t = active_t[keep_local]
                active_region_count = active_region_count[keep_local]
                lambda_vals = lambda_vals[keep_local]
                n_active = active_indices.shape[0]
                
                # 更新 patterns
                new_patterns = []
                for p in state.activation_pattern.patterns:
                    new_patterns.append(p[keep_local])
                state.activation_pattern. patterns = new_patterns
            
            if n_active == 0:
                continue
            
            write_indices = active_region_count
            
            # 记录 entry_t
            entry_ts[active_indices, write_indices] = active_t
            
            # 记录激活模式
            for layer_idx, pattern in enumerate(state.activation_pattern.patterns):
                activation_patterns[layer_idx][active_indices, write_indices] = pattern
            
            # 计算 exit_t
            no_boundary = torch.isinf(lambda_vals) | (lambda_vals <= 0)
            has_boundary = ~no_boundary
            
            # 处理没有边界的样本
            if no_boundary.any():
                no_bd_local = torch.where(no_boundary)[0]
                no_bd_global = active_indices[no_bd_local]
                no_bd_write = write_indices[no_bd_local]
                
                exit_ts[no_bd_global, no_bd_write] = max_distance
                region_count[no_bd_global] += 1
                current_t[no_bd_global] = max_distance
                active_mask[no_bd_global] = False
            
            # 处理有边界的样本
            if has_boundary.any():
                has_bd_local = torch.where(has_boundary)[0]
                has_bd_global = active_indices[has_bd_local]
                has_bd_write = write_indices[has_bd_local]
                has_bd_t = active_t[has_bd_local]
                has_bd_lambda = lambda_vals[has_bd_local]
                
                exit_t_vals = has_bd_t + has_bd_lambda
                
                # 超过最大距离
                exceed_max = exit_t_vals >= max_distance
                if exceed_max.any():
                    exceed_local = torch. where(exceed_max)[0]
                    exceed_global = has_bd_global[exceed_local]
                    exceed_write = has_bd_write[exceed_local]
                    
                    exit_ts[exceed_global, exceed_write] = max_distance
                    region_count[exceed_global] += 1
                    current_t[exceed_global] = max_distance
                    active_mask[exceed_global] = False
                
                # 正常跨越
                normal = ~exceed_max
                if normal.any():
                    normal_local = torch. where(normal)[0]
                    normal_global = has_bd_global[normal_local]
                    normal_write = has_bd_write[normal_local]
                    normal_exit_t = exit_t_vals[normal_local]
                    normal_lambda = has_bd_lambda[normal_local]
                    
                    exit_ts[normal_global, normal_write] = normal_exit_t
                    region_count[normal_global] += 1
                    
                    step = normal_lambda + CROSSING_EPSILON
                    current_x[normal_global] += step. view(-1, *([1] * (current_x.dim() - 1))) * directions[normal_global]
                    current_t[normal_global] += step
            
            # 更新上一步的激活模式
            if prev_pattern_tensors is None:
                prev_pattern_tensors = [
                    torch.zeros((batch_size,) + shape, dtype=torch. bool, device=self.device)
                    for shape in pattern_shapes
                ]
            
            for layer_idx, pattern in enumerate(state.activation_pattern. patterns):
                prev_pattern_tensors[layer_idx][active_indices] = pattern
        
        return BatchTraversalResult(
            batch_size=batch_size,
            num_regions=region_count,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            total_distances=current_t,
            activation_patterns=activation_patterns
        )
    
    def traverse_batch_simple(
        self,
        starts: torch.Tensor,
        directions: torch. Tensor,
        max_distance: float = 10.0,
        max_regions: int = 100,
        normalize_dir: bool = True
    ) -> List[TraversalResult]:
        """批量遍历，返回 TraversalResult 列表"""
        batch_result = self.traverse_batch(
            starts=starts,
            directions=directions,
            max_distance=max_distance,
            max_regions=max_regions,
            normalize_dir=normalize_dir
        )
        
        results = []
        for i in range(batch_result.batch_size):
            results.append(batch_result.get_single_result(i, starts, directions))
        
        return results