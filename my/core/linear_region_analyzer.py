# -*- coding: utf-8 -*-
"""
Simplified Linear Region Analyzer - Only computes 4 core metrics

Core metrics:
1. num_regions: Number of linear regions along a direction
2. mean_gradient_norm: Average gradient norm value
3. mean_gradient_norm_change: Average gradient norm change at boundaries
4. mean_loss_change: Average loss change at boundaries

Design principles:
- Minimize memory footprint (no full activation pattern history)
- Compute only required metrics (no spectral norm, logit changes, etc.)
- Release intermediate tensors promptly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from copy import deepcopy
import numpy as np


# Constants
EPSILON = 1e-6
BOUNDARY_EPS_MIN = 1e-5  # Minimum epsilon for boundary point calculation
BOUNDARY_EPS_RATIO = 0.1  # Ratio of lambda to use as epsilon
MAX_BINARY_SEARCH_ITERATIONS = 20  # Maximum iterations for binary search
TEST_DISTANCES = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Distances to test for boundary


@dataclass
class SimpleAnalysisResult:
    """Simplified analysis result - contains only 4 core metrics"""
    num_regions: int                    # Number of regions
    mean_gradient_norm: float           # Average gradient norm
    mean_gradient_norm_change: float    # Average gradient norm change
    mean_loss_change: float             # Average loss change


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    """Normalize direction vector to unit length"""
    flat = direction.view(direction.shape[0], -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    norm = norm.view((direction.shape[0],) + (1,) * (len(direction.shape) - 1))
    return direction / (norm + 1e-8)


def _calculate_boundary_epsilon(lambda_val: float) -> float:
    """
    Calculate epsilon for boundary point computation.
    
    Args:
        lambda_val: Distance to boundary
        
    Returns:
        Epsilon value for computing before/after boundary points
    """
    return max(BOUNDARY_EPS_MIN, lambda_val * BOUNDARY_EPS_RATIO)


class SimpleLinearRegionAnalyzer:
    """
    Simplified Linear Region Analyzer - Minimizes memory footprint and computation.
    
    Computes only 4 core metrics:
    1. Number of regions
    2. Mean gradient norm
    3. Mean gradient norm change at boundaries
    4. Mean loss change at boundaries
    
    Usage:
        analyzer = SimpleLinearRegionAnalyzer(model, input_shape=(784,), device='cuda')
        result = analyzer.analyze_direction(x, direction, label, max_distance=1.0)
        print(f"Regions: {result.num_regions}")
        print(f"Mean gradient norm: {result.mean_gradient_norm}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu"
    ):
        """
        Args:
            model: PyTorch model (with ReLU activations)
            input_shape: Input shape (without batch dimension)
            device: Computation device
        """
        self.device = device
        self.input_shape = input_shape
        
        # Copy model to avoid modifying original
        self._model = deepcopy(model)
        self._model.eval()
        self._model.requires_grad_(False)
        self._model.to(device)
        
        # Get model dtype
        self._model_dtype = next(self._model.parameters()).dtype
        
        # Find ReLU layers
        self._relu_modules = []
        self._find_relu_layers()
        
        # Pre-create constants to avoid repeated creation
        self._epsilon = torch.tensor([EPSILON], dtype=torch.float64, device=device)
        self._inf = torch.tensor([np.inf], dtype=torch.float64, device=device)
        
    def _find_relu_layers(self):
        """Find all ReLU layers in the model"""
        relu_order = []
        
        def order_hook(module, inp, out):
            relu_order.append(module)
        
        temp_hooks = []
        for module in self._model.modules():
            if isinstance(module, nn.ReLU):
                temp_hooks.append(module.register_forward_hook(order_hook))
        
        # Run one forward pass to determine ReLU order
        dummy = torch.ones((1,) + self.input_shape, device=self.device, dtype=self._model_dtype)
        with torch.no_grad():
            self._model(dummy)
        
        for h in temp_hooks:
            h.remove()
        
        self._relu_modules = relu_order
        self._num_relus = len(self._relu_modules)
    
    def _ensure_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input has batch dimension"""
        if x.dim() == len(self.input_shape):
            return x.unsqueeze(0)
        return x
    
    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        """Transfer to device with correct dtype"""
        return x.to(device=self.device, dtype=self._model_dtype)
    
    def _compute_gradient_norm(self, x: torch.Tensor) -> float:
        """
        Compute gradient norm (Frobenius norm).
        
        Optimized: Accumulates sum of squares directly without constructing full Jacobian.
        """
        x_input = self._ensure_batch_dim(x)
        x_input = self._to_device(x_input)
        
        # Requires gradient
        x_grad = x_input.detach().clone().requires_grad_(True)
        
        logits = self._model(x_grad)
        num_classes = logits.shape[1]
        
        # Compute gradients for all outputs and accumulate squared sum
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
        Batch compute gradient norms.
        
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
        """Compute loss for a single point"""
        x_input = self._ensure_batch_dim(x)
        x_input = self._to_device(x_input)
        
        with torch.no_grad():
            logits = self._model(x_input)
            loss = F.cross_entropy(logits, torch.tensor([label], device=self.device))
        
        return loss.item()
    
    def _compute_loss_batch(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Batch compute losses"""
        points = self._to_device(points)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            logits = self._model(points)
            losses = F.cross_entropy(logits, labels, reduction='none')
        
        return losses
    
    def _get_activation_pattern(self, x: torch.Tensor) -> List[np.ndarray]:
        """
        Get activation pattern (simplified version).
        Stores only boolean values, not full tensors.
        
        Note: Assumes single-input ReLU modules (input[0]).
        """
        patterns = []
        
        def hook(module, input, output):
            # Only record whether activated, convert to numpy to save memory
            # Assumes single-input module (input[0])
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
        """Compare if two activation patterns are identical"""
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
        Compute distance to nearest boundary.
        
        Uses numerical method: detects activation pattern changes.
        """
        x = self._ensure_batch_dim(x)
        direction = self._ensure_batch_dim(direction)
        x = self._to_device(x)
        direction = self._to_device(direction)
        
        # Get current activation pattern
        current_pattern = self._get_activation_pattern(x)
        
        # Binary search to find nearest boundary
        low = 0.0
        high = 1.0
        
        # First find a distance that changes activation pattern
        boundary_found = False
        
        for dist in TEST_DISTANCES:
            x_test = x + dist * direction
            test_pattern = self._get_activation_pattern(x_test)
            if not self._pattern_same(current_pattern, test_pattern):
                high = dist
                boundary_found = True
                break
        
        if not boundary_found:
            return float('inf')
        
        # Binary search for precise boundary
        for _ in range(MAX_BINARY_SEARCH_ITERATIONS):
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
        Analyze linear regions along a given direction.
        
        Core workflow:
        1. Traverse linear regions (only record boundary positions)
        2. Compute gradient norm at region midpoints
        3. Compute gradient norm change before/after boundaries
        4. Compute loss change before/after boundaries (if label provided)
        5. Return statistics
        
        Args:
            x: Starting point (batch=1 or input_shape)
            direction: Direction vector (should be normalized)
            label: Label (for computing loss)
            max_distance: Maximum traversal distance
            max_regions: Maximum number of regions
            
        Returns:
            SimpleAnalysisResult
        """
        x = self._ensure_batch_dim(x)
        direction = self._ensure_batch_dim(direction)
        x = self._to_device(x)
        direction = self._to_device(direction)
        
        # Normalize direction
        direction = normalize_direction(direction)
        
        # Initialize statistics
        num_regions = 0
        gradient_norms = []
        gradient_changes = []
        loss_changes = []
        
        current_t = 0.0
        
        # Get initial activation pattern
        prev_pattern = self._get_activation_pattern(x)
        
        while current_t < max_distance and num_regions < max_regions:
            # Compute current position
            current_x = x + current_t * direction
            
            # Compute distance to next boundary
            lambda_val = self._compute_lambda_to_boundary(current_x, direction)
            
            if lambda_val <= 0 or lambda_val == float('inf'):
                # Last region or cannot find boundary
                num_regions += 1
                
                # Compute gradient norm at region midpoint
                remaining_dist = max_distance - current_t
                mid_t = current_t + remaining_dist / 2
                mid_x = x + mid_t * direction
                grad_norm = self._compute_gradient_norm(mid_x)
                gradient_norms.append(grad_norm)
                break
            
            # Adjust lambda to not exceed max_distance
            actual_lambda = min(lambda_val, max_distance - current_t)
            
            # Record region
            num_regions += 1
            
            # Gradient norm at region midpoint
            mid_t = current_t + actual_lambda / 2
            mid_x = x + mid_t * direction
            grad_norm = self._compute_gradient_norm(mid_x)
            gradient_norms.append(grad_norm)
            
            # If actual_lambda equals remaining distance, we've reached the end
            if actual_lambda >= max_distance - current_t - EPSILON:
                break
            
            # Gradient and loss before/after boundary
            boundary_t = current_t + actual_lambda
            eps = _calculate_boundary_epsilon(actual_lambda)
            
            x_before = x + (boundary_t - eps) * direction
            x_after = x + (boundary_t + eps) * direction
            
            grad_before = self._compute_gradient_norm(x_before)
            grad_after = self._compute_gradient_norm(x_after)
            gradient_changes.append(abs(grad_after - grad_before))
            
            if label is not None:
                loss_before = self._compute_loss(x_before, label)
                loss_after = self._compute_loss(x_after, label)
                loss_changes.append(loss_after - loss_before)
            
            # Move to next region
            current_t = boundary_t + eps
            
            # Check if activation pattern changed
            new_x = x + current_t * direction
            new_pattern = self._get_activation_pattern(new_x)
            if self._pattern_same(prev_pattern, new_pattern):
                # Pattern didn't change, possibly numerical issue, continue trying
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
        Batch analysis - Collects all points then computes in batch.
        
        Core approach:
        1. Serially traverse all samples, collect points for gradient computation
        2. Merge all points into a large batch
        3. Compute all gradient norms at once
        4. Distribute results back to each sample
        
        Args:
            x_batch: Input batch (batch, *input_shape)
            directions: Direction batch (batch, *input_shape)
            labels: Label batch (batch,)
            max_distance: Maximum traversal distance
            max_regions: Maximum number of regions
            
        Returns:
            List[SimpleAnalysisResult]
        """
        batch_size = x_batch.shape[0]
        x_batch = self._to_device(x_batch)
        directions = self._to_device(directions)
        directions = normalize_direction(directions)
        
        # Step 1: Traverse and collect all points to compute
        all_mid_points = []
        all_boundary_before = []
        all_boundary_after = []
        
        point_mapping = []  # [(sample_idx, num_mid_points, num_boundaries), ...]
        region_counts = []
        
        for i in range(batch_size):
            x = x_batch[i:i+1]
            direction = directions[i:i+1]
            label = labels[i].item() if labels is not None else None
            
            # Traverse this sample and collect points
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
        
        # Step 2: Batch compute all gradient norms
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
        
        # Step 3: Batch compute losses (if needed)
        boundary_loss_before = []
        boundary_loss_after = []
        if labels is not None and len(all_boundary_before) > 0:
            # Prepare labels for each boundary point
            boundary_labels = []
            for i, mapping in enumerate(point_mapping):
                boundary_labels.extend([labels[i].item()] * mapping['num_boundary'])
            
            if len(boundary_labels) > 0:
                boundary_labels_tensor = torch.tensor(boundary_labels, dtype=torch.long, device=self.device)
                boundary_loss_before = self._compute_loss_batch(before_tensor, boundary_labels_tensor)
                boundary_loss_after = self._compute_loss_batch(after_tensor, boundary_labels_tensor)
        
        # Step 4: Distribute results back to each sample
        results = []
        mid_idx = 0
        boundary_idx = 0
        
        for i, mapping in enumerate(point_mapping):
            num_mid = mapping['num_mid']
            num_boundary = mapping['num_boundary']
            
            # Extract gradient norms for this sample
            if num_mid > 0:
                sample_grad_norms = mid_grad_norms[mid_idx:mid_idx + num_mid].tolist()
                mean_grad_norm = sum(sample_grad_norms) / len(sample_grad_norms)
            else:
                mean_grad_norm = 0.0
            
            # Extract gradient changes for this sample
            if num_boundary > 0:
                before = boundary_grad_before[boundary_idx:boundary_idx + num_boundary]
                after = boundary_grad_after[boundary_idx:boundary_idx + num_boundary]
                grad_changes = torch.abs(after - before).tolist()
                mean_grad_change = sum(grad_changes) / len(grad_changes)
            else:
                mean_grad_change = 0.0
            
            # Extract loss changes for this sample
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
        Traverse linear regions and collect points for computation.
        
        Returns:
            dict with keys:
                - num_regions: int
                - mid_points: List[Tensor] - midpoint of each region
                - boundary_before: List[Tensor] - points before boundary
                - boundary_after: List[Tensor] - points after boundary
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
            eps = _calculate_boundary_epsilon(actual_lambda)
            
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
        Compute direction toward decision boundary.
        
        Direction is defined as: ∇_x (logit[top2] - logit[top1])
        Moving in this direction reduces the gap between top1 and top2.
        
        Args:
            x: Input data point (batch, *input_shape)
            normalize: Whether to normalize direction vector
            
        Returns:
            direction: Direction vector
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
        Convenience method: Automatically compute decision boundary direction and analyze.
        
        Args:
            x: Input point
            label: Label
            max_distance: Maximum distance
            max_regions: Maximum number of regions
            
        Returns:
            SimpleAnalysisResult
        """
        x = self._ensure_batch_dim(x)
        x = self._to_device(x)
        
        direction = self.find_decision_boundary_direction(x)
        
        return self.analyze_direction(x, direction, label, max_distance, max_regions)
    
    def cleanup(self):
        """Clean up resources (no special cleanup needed in current implementation)"""
        pass
