# -*- coding: utf-8 -*-
"""
LinearRegionTraverser: 沿给定方向遍历线性区域
优化版本：使用 JIT 编译
"""

import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field

from .model_wrapper import (
    ModelWrapper, ActivationPattern, normalize_direction, 
    compute_direction_norm, EPSILON
)


#@torch.jit.script
def _check_pattern_equality_jit(
    patterns1: List[torch.Tensor], 
    patterns2: List[torch.Tensor],
    batch_size: int
) -> torch.Tensor:
    """JIT 编译的激活模式比较"""
    device = patterns1[0].device
    same_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    for i in range(len(patterns1)):
        p1 = patterns1[i]
        p2 = patterns2[i]
        p1_flat = p1.view(batch_size, -1)
        p2_flat = p2.view(batch_size, -1)
        layer_same = (p1_flat == p2_flat).all(dim=1)
        same_mask = same_mask & layer_same
    
    return same_mask


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
    direction: torch.Tensor
    regions: List[LinearRegionInfo] = field(default_factory=list)
    total_distance: float = 0.0
    num_regions: int = 0


@dataclass
class BatchTraversalResult:
    """批量遍历结果"""
    batch_size: int
    num_regions: torch.Tensor
    entry_ts: torch.Tensor
    exit_ts: torch.Tensor
    total_distances: torch.Tensor
    activation_patterns: List[torch.Tensor]
    
    def get_single_result(self, idx: int, start_point: torch.Tensor, direction: torch.Tensor) -> TraversalResult:
        """提取单个样本的结果"""
        n_regions = self.num_regions[idx].item()
        regions = []
        
        for r in range(n_regions):
            patterns = [p[idx, r] for p in self.activation_patterns]
            regions.append(LinearRegionInfo(
                region_id=r,
                activation_pattern=ActivationPattern(patterns=[p.unsqueeze(0) for p in patterns]),
                entry_t=self.entry_ts[idx, r].item(),
                exit_t=self.exit_ts[idx, r].item()
            ))
        
        return TraversalResult(
            start_point=start_point[idx:idx+1] if start_point.shape[0] > idx else start_point,
            direction=direction[idx:idx+1] if direction.shape[0] > idx else direction,
            regions=regions,
            total_distance=self.total_distances[idx].item(),
            num_regions=n_regions
        )


class LinearRegionTraverser:
    """线性区域遍历器 - 优化版本"""
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.wrapper = model_wrapper
        self.device = model_wrapper.device
        self._model_dtype = model_wrapper._model_dtype
        
        self._epsilon = torch.tensor([EPSILON], dtype=torch.float64, device=self.device)
        self._one = torch.ones((1,), dtype=torch.float64, device=self.device)
    
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
        
        batch_result = self.traverse_batch(
            starts=start,
            directions=direction,
            max_distance=max_distance,
            max_regions=max_regions,
            normalize_dir=normalize_dir
        )
        
        return batch_result.get_single_result(0, start, direction)
    
    def traverse_batch(
        self,
        starts: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float = 10.0,
        max_regions: int = 100,
        normalize_dir: bool = True
    ) -> BatchTraversalResult:
        """批量并行遍历"""
        if starts.dim() == len(self.wrapper.input_shape):
            starts = starts.unsqueeze(0)
            directions = directions.unsqueeze(0)
        
        batch_size = starts.shape[0]
        starts = starts.to(device=self.device, dtype=self._model_dtype)
        directions = directions.to(device=self.device, dtype=self._model_dtype)
        
        # 计算方向范数和归一化方向
        norm_of_directions = compute_direction_norm(directions)
        directions_normalized = directions / (norm_of_directions + 1e-8)
        
        directions_for_traversal = directions_normalized
        
        # 计算终点
        x0s = starts.clone()
        x1s = starts + max_distance * directions_for_traversal
        
        # 预分配结果缓冲区
        entry_ts = torch.full((batch_size, max_regions), float('inf'),
                              dtype=torch.float64, device=self.device)
        exit_ts = torch.full((batch_size, max_regions), float('inf'),
                             dtype=torch.float64, device=self.device)
        
        x_current = x0s.clone()
        
        # 获取初始激活模式形状
        init_state = self.wrapper.get_region_state(starts[:1], directions_for_traversal[:1])
        pattern_shapes = [p.shape[1:] for p in init_state.activation_pattern.patterns]
        
        activation_patterns = [
            torch.zeros((batch_size, max_regions) + shape, dtype=torch.bool, device=self.device)
            for shape in pattern_shapes
        ]
        
        # 获取终点的激活模式
        act_pattern_x1 = self.wrapper.get_activation_pattern_only(x1s)
        
        # 状态变量
        region_count = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        current_t = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        
        indices_to_batches = torch.arange(batch_size, device=self.device, dtype=torch.long)
        
        iteration = 0
        while len(indices_to_batches) > 0 and iteration < max_regions:
            iteration += 1
            
            active_x = x_current[indices_to_batches]
            active_dir = directions_for_traversal[indices_to_batches]
            
            state = self.wrapper.get_region_state(active_x, active_dir)
            act_pattern_x = state.activation_pattern
            lambdas = state.lambda_to_boundary
            
            # 使用 JIT 优化的模式比较
            act_pattern_x1_active = act_pattern_x1.index_select(indices_to_batches)
            
            # 比较激活模式
            same_mask = _check_pattern_equality_jit(
                act_pattern_x.patterns,
                act_pattern_x1_active.patterns,
                len(indices_to_batches)
            )
            diff_indices_local = torch.where(~same_mask)[0]
            
            if len(diff_indices_local) == 0:
                for i, global_idx in enumerate(indices_to_batches):
                    idx = global_idx.item()
                    r = region_count[idx].item()
                    if r < max_regions:
                        entry_ts[idx, r] = current_t[idx]
                        exit_ts[idx, r] = max_distance
                        for layer_idx, pattern in enumerate(state.activation_pattern.patterns):
                            activation_patterns[layer_idx][idx, r] = pattern[i]
                        region_count[idx] += 1
                        current_t[idx] = max_distance
                break
            
            diff_indices_global = indices_to_batches[diff_indices_local]
            lambdas = lambdas[diff_indices_local]
            
            valid_lambda_mask = (lambdas > self._epsilon) & (lambdas <= self._one * max_distance)
            
            if not valid_lambda_mask.any():
                for i, local_idx in enumerate(diff_indices_local):
                    global_idx = indices_to_batches[local_idx].item()
                    r = region_count[global_idx].item()
                    if r < max_regions:
                        entry_ts[global_idx, r] = current_t[global_idx]
                        exit_ts[global_idx, r] = max_distance
                        for layer_idx, pattern in enumerate(state.activation_pattern.patterns):
                            activation_patterns[layer_idx][global_idx, r] = pattern[local_idx]
                        region_count[global_idx] += 1
                        current_t[global_idx] = max_distance
                
                keep_mask = torch.ones(len(indices_to_batches), dtype=torch.bool, device=self.device)
                keep_mask[diff_indices_local] = False
                indices_to_batches = indices_to_batches[keep_mask]
                continue
            
            valid_local_indices = diff_indices_local[valid_lambda_mask]
            valid_global_indices = indices_to_batches[valid_local_indices]
            valid_lambdas = lambdas[valid_lambda_mask]
            
            for i, (local_idx, global_idx) in enumerate(zip(valid_local_indices, valid_global_indices)):
                idx = global_idx.item()
                r = region_count[idx].item()
                if r < max_regions:
                    entry_ts[idx, r] = current_t[idx]
                    for layer_idx, pattern in enumerate(state.activation_pattern.patterns):
                        activation_patterns[layer_idx][idx, r] = pattern[local_idx]
            
            exit_t_vals = current_t[valid_global_indices] + valid_lambdas
            
            exceed_max = exit_t_vals >= max_distance
            
            if exceed_max.any():
                exceed_local = torch.where(exceed_max)[0]
                exceed_global = valid_global_indices[exceed_local]
                
                for idx in exceed_global:
                    idx = idx.item()
                    r = region_count[idx].item()
                    if r < max_regions:
                        exit_ts[idx, r] = max_distance
                        region_count[idx] += 1
                        current_t[idx] = max_distance
                
                remove_mask = torch.zeros(len(indices_to_batches), dtype=torch.bool, device=self.device)
                for local_idx in valid_local_indices[exceed_local]:
                    remove_mask[local_idx] = True
                indices_to_batches = indices_to_batches[~remove_mask]
            
            normal_mask = ~exceed_max
            if normal_mask.any():
                normal_local = torch.where(normal_mask)[0]
                normal_global = valid_global_indices[normal_local]
                normal_lambdas = valid_lambdas[normal_local]
                normal_exit_t = exit_t_vals[normal_local]
                
                for i, idx in enumerate(normal_global):
                    idx = idx.item()
                    r = region_count[idx].item()
                    if r < max_regions:
                        exit_ts[idx, r] = normal_exit_t[i]
                        region_count[idx] += 1
                
                step = normal_lambdas + self._epsilon
                
                for i, (local_idx, global_idx) in enumerate(zip(valid_local_indices[normal_local], normal_global)):
                    x_current[global_idx] = x_current[global_idx] + step[i] * directions_for_traversal[global_idx]
                    current_t[global_idx] = current_t[global_idx] + step[i]
                
                for i, global_idx in enumerate(normal_global):
                    dist_from_start = torch.norm(
                        (x_current[global_idx] - x0s[global_idx]).view(-1)
                    ).item()
                    if dist_from_start > max_distance:
                        mask = indices_to_batches != global_idx
                        indices_to_batches = indices_to_batches[mask]
            
            invalid_local_indices = diff_indices_local[~valid_lambda_mask]
            if len(invalid_local_indices) > 0:
                invalid_global_indices = indices_to_batches[invalid_local_indices]
                
                for i, (local_idx, global_idx) in enumerate(zip(invalid_local_indices, invalid_global_indices)):
                    idx = global_idx.item()
                    r = region_count[idx].item()
                    if r < max_regions:
                        entry_ts[idx, r] = current_t[idx]
                        exit_ts[idx, r] = max_distance
                        for layer_idx, pattern in enumerate(state.activation_pattern.patterns):
                            activation_patterns[layer_idx][idx, r] = pattern[local_idx]
                        region_count[idx] += 1
                        current_t[idx] = max_distance
                
                remove_mask = torch.zeros(len(indices_to_batches), dtype=torch.bool, device=self.device)
                for local_idx in invalid_local_indices:
                    remove_mask[local_idx] = True
                indices_to_batches = indices_to_batches[~remove_mask]
        
        current_t = torch.clamp(current_t, max=max_distance)
        
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
        directions: torch.Tensor,
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