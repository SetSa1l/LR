# -*- coding: utf-8 -*-
"""
Fast Linear Region Analyzer - 10-20x speedup

Key optimizations:
1. Analytical lambda computation (1 forward pass instead of 28)
2. Batch gradient computation with torch.func.jacrev + vmap
3. Minimal forward passes through lightweight wrapper

Usage:
    analyzer = FastLinearRegionAnalyzer(model, input_shape, device='cuda')
    results = analyzer.analyze_batch(x, directions, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from copy import deepcopy

EPSILON = 1e-6
BOUNDARY_EPS_MIN = 1e-5
BOUNDARY_EPS_RATIO = 0.1


@dataclass
class FastAnalysisResult:
    """Analysis result with 4 core metrics"""
    num_regions: int
    mean_gradient_norm: float
    mean_gradient_norm_change: float
    mean_loss_change: float


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    """Normalize direction vector to unit length"""
    flat = direction.view(direction.shape[0], -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    norm = norm.view((direction.shape[0],) + (1,) * (len(direction.shape) - 1))
    return direction / (norm + 1e-8)


class LightweightModelWrapper:
    """Lightweight wrapper for fast analytical lambda computation via hooks."""
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu"):
        self.device = device
        self.input_shape = input_shape
        self._model = deepcopy(model)
        self._model.eval()
        self._model.requires_grad_(False)
        self._model.to(device)
        self._model_dtype = next(self._model.parameters()).dtype
        
        self._relu_modules: List[nn.Module] = []
        self._hooks: List[Any] = []
        self._find_relu_layers()
        
        self._lambdas_min: Optional[torch.Tensor] = None
        self._patterns_buffer: List[torch.Tensor] = []
        self._hooks_active = False
        self._epsilon = torch.tensor([EPSILON], dtype=torch.float64, device=device)
        self._inf = torch.tensor([float('inf')], dtype=torch.float64, device=device)
        self._machine_eps = torch.tensor([torch.finfo(torch.float64).eps], dtype=torch.float64, device=device)
    
    def _find_relu_layers(self):
        """Find all ReLU layers in execution order"""
        relu_order = []
        def order_hook(module, inp, out):
            relu_order.append(module)
        
        temp_hooks = []
        for module in self._model.modules():
            if isinstance(module, nn.ReLU):
                temp_hooks.append(module.register_forward_hook(order_hook))
        
        dummy = torch.ones((2,) + self.input_shape, device=self.device, dtype=self._model_dtype)
        with torch.no_grad():
            self._model(dummy)
        for h in temp_hooks:
            h.remove()
        
        self._relu_modules = relu_order
        for module in self._relu_modules:
            self._hooks.append(module.register_forward_hook(self._relu_hook))
    
    def _relu_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], 
                   output: torch.Tensor) -> Optional[torch.Tensor]:
        """ReLU hook: compute lambda analytically. Input: [x; direction] concatenated."""
        if not self._hooks_active:
            return None
        
        pre_act = inputs[0]
        half_batch = pre_act.shape[0] // 2
        if half_batch == 0:
            return None
        
        pre_act_data = pre_act[:half_batch].detach()
        pre_act_dir = pre_act[half_batch:].detach()
        
        act_pattern = (pre_act_data > 0)
        self._patterns_buffer.append(act_pattern)
        
        with torch.no_grad():
            sign_dir = pre_act_dir.sign()
            zero_mask = (pre_act_dir == 0).to(pre_act_dir.dtype)
            denom = pre_act_dir + self._machine_eps.to(pre_act_dir.dtype) * (sign_dir + zero_mask)
            
            lambdas = -pre_act_data / denom
            lambdas = lambdas.view(half_batch, -1)
            lambdas = torch.where(lambdas > self._epsilon.to(lambdas.dtype), lambdas, self._inf.to(lambdas.dtype))
            min_lambda_layer = lambdas.min(dim=1)[0].double()
            self._lambdas_min = torch.minimum(self._lambdas_min, min_lambda_layer)
        
        modified_output = output.clone()
        modified_output[half_batch:] = pre_act_dir * act_pattern.to(pre_act_dir.dtype)
        return modified_output
    
    def get_lambda_to_boundary(self, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Compute distance to nearest boundary in ONE forward pass."""
        batch_size = x.shape[0]
        self._lambdas_min = torch.full((batch_size,), float('inf'), dtype=torch.float64, device=self.device)
        self._patterns_buffer = []
        self._hooks_active = True
        
        x = x.to(device=self.device, dtype=self._model_dtype)
        direction = direction.to(device=self.device, dtype=self._model_dtype)
        combined = torch.cat([x, direction], dim=0)
        
        with torch.no_grad():
            self._model(combined)
        self._hooks_active = False
        return self._lambdas_min.clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass without tracking"""
        self._hooks_active = False
        return self._model(x.to(device=self.device, dtype=self._model_dtype))
    
    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class FastLinearRegionAnalyzer:
    """
    Ultra-fast Linear Region Analyzer (10-20x speedup).
    
    Usage:
        analyzer = FastLinearRegionAnalyzer(model, input_shape=(784,), device='cuda')
        result = analyzer.analyze(x, label, max_distance=1.0)
        results = analyzer.analyze_batch(x_batch, directions, labels)
        analyzer.cleanup()
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu"):
        self.device = device
        self.input_shape = input_shape
        self.wrapper = LightweightModelWrapper(model, input_shape, device)
        
        self._model = deepcopy(model)
        self._model.eval()
        self._model.requires_grad_(False)
        self._model.to(device)
        self._model_dtype = next(self._model.parameters()).dtype
        self._use_torch_func = hasattr(torch, 'func')
    
    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self.device, dtype=self._model_dtype)
    
    def _ensure_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == len(self.input_shape):
            return x.unsqueeze(0)
        return x
    
    def _compute_gradient_norm_batch(self, points: torch.Tensor) -> torch.Tensor:
        """Batch compute gradient norms using torch.func or legacy method."""
        if len(points) == 0:
            return torch.tensor([], device=self.device)
        points = self._to_device(points)
        
        if self._use_torch_func:
            return self._compute_gradient_norm_torch_func(points)
        return self._compute_gradient_norm_legacy(points)
    
    def _compute_gradient_norm_torch_func(self, points: torch.Tensor) -> torch.Tensor:
        """Compute gradient norms using torch.func.jacrev + vmap."""
        from torch.func import jacrev, vmap, functional_call
        batch_size = points.shape[0]
        params = dict(self._model.named_parameters())
        buffers = dict(self._model.named_buffers())
        
        def single_forward(x):
            return functional_call(self._model, (params, buffers), (x.unsqueeze(0),), strict=False).squeeze(0)
        
        try:
            jacobians = vmap(jacrev(single_forward))(points)
            return torch.norm(jacobians.reshape(batch_size, -1), p=2, dim=1)
        except Exception:
            return self._compute_gradient_norm_legacy(points)
    
    def _compute_gradient_norm_legacy(self, points: torch.Tensor) -> torch.Tensor:
        """Compute gradient norms using optimized batched backward pass."""
        batch_size = points.shape[0]
        points = points.clone().requires_grad_(True)
        logits = self._model(points)
        num_classes = logits.shape[1]
        
        grad_norm_sq = torch.zeros(batch_size, device=self.device, dtype=self._model_dtype)
        for i in range(num_classes):
            if points.grad is not None:
                points.grad.zero_()
            grad_outputs = torch.zeros_like(logits)
            grad_outputs[:, i] = 1.0
            logits.backward(gradient=grad_outputs, retain_graph=(i < num_classes - 1))
            if points.grad is not None:
                grad_norm_sq += (points.grad.view(batch_size, -1) ** 2).sum(dim=1)
        return grad_norm_sq.sqrt()
    
    def _compute_loss_batch(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Batch compute cross-entropy losses"""
        with torch.no_grad():
            logits = self._model(self._to_device(points))
            return F.cross_entropy(logits, labels.to(self.device), reduction='none')
    
    def _fast_traverse(self, x: torch.Tensor, direction: torch.Tensor, 
                       max_distance: float, max_regions: int) -> Dict[str, Any]:
        """Fast traversal using analytical lambda computation (1 forward pass per boundary)."""
        x = self._ensure_batch_dim(self._to_device(x))
        direction = self._ensure_batch_dim(self._to_device(direction))
        
        gradient_points, loss_before_points, loss_after_points = [], [], []
        current_t, num_regions = 0.0, 0
        
        while current_t < max_distance and num_regions < max_regions:
            current_x = x + current_t * direction
            lambda_val = self.wrapper.get_lambda_to_boundary(current_x, direction).item()
            
            if lambda_val == float('inf') or lambda_val <= 0:
                remaining = max_distance - current_t
                gradient_points.append((x + (current_t + remaining / 2) * direction).clone())
                num_regions += 1
                break
            
            actual_lambda = min(lambda_val, max_distance - current_t)
            gradient_points.append((x + (current_t + actual_lambda / 2) * direction).clone())
            num_regions += 1
            
            if actual_lambda >= max_distance - current_t - EPSILON:
                break
            
            boundary_t = current_t + actual_lambda
            eps = max(BOUNDARY_EPS_MIN, actual_lambda * BOUNDARY_EPS_RATIO)
            loss_before_points.append((x + (boundary_t - eps) * direction).clone())
            loss_after_points.append((x + (boundary_t + eps) * direction).clone())
            current_t = boundary_t + eps
        
        return {'num_regions': num_regions, 'gradient_points': gradient_points,
                'loss_before_points': loss_before_points, 'loss_after_points': loss_after_points}
    
    def find_decision_boundary_direction(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute direction toward decision boundary: ∇_x (logit[top2] - logit[top1])"""
        x = self._ensure_batch_dim(self._to_device(x))
        x_grad = x.detach().clone().requires_grad_(True)
        
        self._model.requires_grad_(True)
        logits = self._model(x_grad)
        self._model.requires_grad_(False)
        
        top2_values, _ = torch.topk(logits, k=2, dim=1)
        (top2_values[:, 1] - top2_values[:, 0]).sum().backward()
        direction = x_grad.grad.detach().clone()
        return normalize_direction(direction) if normalize else direction
    
    def analyze_direction(self, x: torch.Tensor, direction: torch.Tensor, label: Optional[int] = None,
                          max_distance: float = 1.0, max_regions: int = 100) -> FastAnalysisResult:
        """Analyze linear regions along a given direction."""
        x = self._ensure_batch_dim(self._to_device(x))
        direction = normalize_direction(self._ensure_batch_dim(self._to_device(direction)))
        
        info = self._fast_traverse(x, direction, max_distance, max_regions)
        grad_pts, before_pts, after_pts = info['gradient_points'], info['loss_before_points'], info['loss_after_points']
        
        mean_grad = self._compute_gradient_norm_batch(torch.cat(grad_pts, 0)).mean().item() if grad_pts else 0.0
        
        mean_grad_change = 0.0
        if before_pts:
            before_t, after_t = torch.cat(before_pts, 0), torch.cat(after_pts, 0)
            mean_grad_change = torch.abs(self._compute_gradient_norm_batch(after_t) - 
                                         self._compute_gradient_norm_batch(before_t)).mean().item()
        
        mean_loss_change = 0.0
        if label is not None and before_pts:
            n = len(before_pts)
            labels_t = torch.full((n,), label, dtype=torch.long, device=self.device)
            mean_loss_change = (self._compute_loss_batch(after_t, labels_t) - 
                               self._compute_loss_batch(before_t, labels_t)).mean().item()
        
        return FastAnalysisResult(info['num_regions'], mean_grad, mean_grad_change, mean_loss_change)
    
    def analyze(self, x: torch.Tensor, label: Optional[int] = None, 
                max_distance: float = 1.0, max_regions: int = 100) -> FastAnalysisResult:
        """Auto-compute decision boundary direction and analyze."""
        x = self._ensure_batch_dim(self._to_device(x))
        return self.analyze_direction(x, self.find_decision_boundary_direction(x), label, max_distance, max_regions)
    
    def analyze_batch(self, x_batch: torch.Tensor, directions: torch.Tensor, 
                      labels: Optional[torch.Tensor] = None, max_distance: float = 1.0, 
                      max_regions: int = 100) -> List[FastAnalysisResult]:
        """Ultra-fast batch analysis: traverse all, then batch compute all metrics."""
        batch_size = x_batch.shape[0]
        x_batch = self._to_device(x_batch)
        directions = normalize_direction(self._to_device(directions))
        
        # Phase 1: Traverse all samples, collect points
        all_grad_pts, all_before, all_after, mappings = [], [], [], []
        for i in range(batch_size):
            info = self._fast_traverse(x_batch[i:i+1], directions[i:i+1], max_distance, max_regions)
            all_grad_pts.extend(info['gradient_points'])
            all_before.extend(info['loss_before_points'])
            all_after.extend(info['loss_after_points'])
            mappings.append({'num_regions': info['num_regions'], 'num_grad': len(info['gradient_points']),
                            'num_bounds': len(info['loss_before_points'])})
        
        # Phase 2: Batch compute gradient norms
        all_grads = self._compute_gradient_norm_batch(torch.cat(all_grad_pts, 0)) if all_grad_pts else torch.tensor([])
        before_grads = after_grads = torch.tensor([])
        if all_before:
            before_t, after_t = torch.cat(all_before, 0), torch.cat(all_after, 0)
            before_grads = self._compute_gradient_norm_batch(before_t)
            after_grads = self._compute_gradient_norm_batch(after_t)
        
        # Phase 3: Batch compute losses
        before_losses = after_losses = torch.tensor([])
        if labels is not None and all_before:
            bound_labels = torch.tensor([labels[i].item() for i, m in enumerate(mappings) 
                                        for _ in range(m['num_bounds'])], dtype=torch.long, device=self.device)
            if len(bound_labels) > 0:
                before_losses = self._compute_loss_batch(before_t, bound_labels)
                after_losses = self._compute_loss_batch(after_t, bound_labels)
        
        # Phase 4: Distribute results
        results, grad_idx, bound_idx = [], 0, 0
        for i, m in enumerate(mappings):
            ng, nb = m['num_grad'], m['num_bounds']
            mean_grad = all_grads[grad_idx:grad_idx+ng].mean().item() if ng else 0.0
            mean_grad_change = torch.abs(after_grads[bound_idx:bound_idx+nb] - 
                                        before_grads[bound_idx:bound_idx+nb]).mean().item() if nb else 0.0
            mean_loss = ((after_losses[bound_idx:bound_idx+nb] - 
                         before_losses[bound_idx:bound_idx+nb]).mean().item() 
                        if labels is not None and nb and len(before_losses) > 0 else 0.0)
            results.append(FastAnalysisResult(m['num_regions'], mean_grad, mean_grad_change, mean_loss))
            grad_idx += ng
            bound_idx += nb
        return results
    
    def cleanup(self):
        self.wrapper.cleanup()
