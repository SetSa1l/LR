# -*- coding: utf-8 -*-
"""
ModelWrapper: 封装 ReLU 网络，追踪激活模式并计算到边界的距离
优化版本：使用 JIT 编译和减少内存拷贝
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy

# Gamba 风格的常量
EPSILON = 1e-6


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    """归一化方向向量"""
    flat = direction.view(direction.shape[0], -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    norm = norm.view((direction.shape[0],) + (1,) * (len(direction.shape) - 1))
    return direction / (norm + 1e-8)


@torch.jit.script
def compute_direction_norm(direction: torch.Tensor) -> torch.Tensor:
    """计算方向向量的范数 - JIT 优化"""
    flat = direction.view(direction.shape[0], -1)
    norm = torch.norm(flat, p=2, dim=1, keepdim=True)
    shape = [direction.shape[0]] + [1] * (len(direction.shape) - 1)
    return norm.view(shape)


@torch.jit.script
def _compute_lambdas_jit(
    pre_act_data: torch.Tensor,
    pre_act_dir: torch.Tensor,
    epsilon: torch.Tensor,
    inf: torch.Tensor,
    machine_eps: torch.Tensor
) -> torch.Tensor:
    """JIT 编译的 lambda 计算 - 核心热点函数"""
    half_batch = pre_act_data.shape[0]
    
    pre_data_f64 = pre_act_data.double()
    pre_dir_f64 = pre_act_dir.double()
    
    # 避免除零
    sign_dir = pre_dir_f64.sign()
    zero_mask = (pre_dir_f64 == 0).float()
    denom = pre_dir_f64 + machine_eps * (sign_dir + zero_mask)
    
    lambdas = -pre_data_f64 / denom
    lambdas = lambdas.view(half_batch, -1)
    
    # 只保留正的 lambda
    lambdas = torch.where(lambdas <= epsilon, inf, lambdas)
    
    min_lambda, _ = lambdas.min(dim=1)
    return min_lambda


@dataclass
class ActivationPattern:
    """存储网络的激活模式"""
    patterns: List[torch.Tensor] = field(default_factory=list)
    
    def equals(self, other: "ActivationPattern") -> torch.Tensor:
        """批量比较激活模式是否相同"""
        if len(self.patterns) == 0:
            raise ValueError("Empty activation pattern")
        
        batch_size = self.patterns[0].shape[0]
        device = self.patterns[0].device
        same_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for p1, p2 in zip(self.patterns, other.patterns):
            p1_flat = p1.view(batch_size, -1)
            p2_flat = p2.view(batch_size, -1)
            layer_same = (p1_flat == p2_flat).all(dim=1)
            same_mask = same_mask & layer_same
        
        return same_mask
    
    def not_equal_indices(self, other: "ActivationPattern") -> torch.Tensor:
        """返回激活模式不同的样本索引"""
        same_mask = self.equals(other)
        return torch.where(~same_mask)[0]
    
    def copy(self):
        return ActivationPattern(patterns=[p.clone() for p in self.patterns])
    
    def index_select(self, indices: torch.Tensor) -> "ActivationPattern":
        """根据索引选择子集"""
        return ActivationPattern(patterns=[p[indices] for p in self.patterns])


@dataclass 
class RegionState:
    """线性区域的状态信息"""
    activation_pattern: ActivationPattern
    lambdas_per_layer: torch.Tensor
    lambda_to_boundary: torch.Tensor
    boundary_layer_idx: torch.Tensor
    logits: torch.Tensor


def is_affine_layer(module: nn.Module) -> bool:
    """判断模块是否是仿射层"""
    return isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d))


class AffineLayerWrapper(nn.Module):
    """包装线性层，分离 bias 计算"""
    
    def __init__(self, linear_module: nn.Module, batch_size: int):
        super().__init__()
        
        if not is_affine_layer(linear_module):
            raise ValueError("Unsupported module type: %s" % type(linear_module))
        
        self.linear = linear_module
        self.half_batch = batch_size
        self.batch_size = batch_size * 2
        self._retain_bias = False
        self._enabled = True
        
        self.bias = None
        self.bias_orig = None
        self._init_bias()
    
    def _init_bias(self):
        if self.linear.bias is None:
            return
        
        bias_orig = self.linear.bias.detach().clone()
        
        if isinstance(self.linear, nn.Linear):
            weight_dims = 0
        elif isinstance(self.linear, (nn.Conv2d, nn.BatchNorm2d)):
            weight_dims = 2
        else:
            weight_dims = 0
        
        shape = (1, -1) + (1,) * weight_dims
        self.bias_orig = nn.Parameter(bias_orig.reshape(shape), requires_grad=False)
        
        if isinstance(self.linear, nn.BatchNorm2d):
            self.bias = self._create_bias_bn(bias_orig)
        else:
            self.bias = self._create_bias_normal(bias_orig)
        
        self.linear.bias = None
    
    def _create_bias_normal(self, bias_orig: torch.Tensor) -> nn.Parameter:
        if isinstance(self.linear, nn.Linear):
            output_shape = (self.batch_size, self.linear.weight.shape[0])
            bias_shape = (1, -1)
        else:
            output_shape = (self.batch_size, self.linear.weight. shape[0], 1, 1)
            bias_shape = (1, -1, 1, 1)
        
        bias_broadcast = torch.zeros(
            output_shape,
            dtype=self.linear.weight.dtype,
            device=self.linear.weight.device
        )
        bias_broadcast[:self.half_batch] = bias_orig.reshape(bias_shape)
        
        return nn.Parameter(bias_broadcast, requires_grad=False)
    
    def _create_bias_bn(self, bias_orig: torch.Tensor) -> nn.Parameter:
        mean = self.linear.running_mean
        var = self.linear.running_var
        eps = self.linear.eps
        weight = self.linear. weight
        
        output_shape = (self.batch_size, bias_orig.shape[0], 1, 1)
        bias_shape = (1, bias_orig.shape[0], 1, 1)
        
        bias_broadcast = torch.zeros(
            output_shape,
            dtype=weight.dtype,
            device=weight.device
        )
        
        bias_broadcast[:self.half_batch] = bias_orig.reshape(bias_shape)
        compensation = (mean / torch.sqrt(var + eps)) * weight
        bias_broadcast[self.half_batch:] = compensation.reshape(bias_shape)
        
        return nn.Parameter(bias_broadcast, requires_grad=False)
    
    def retain_bias_(self, retain: bool = False):
        self._retain_bias = retain and (self.bias_orig is not None)
    
    def set_enabled(self, enabled: bool):
        self._enabled = enabled
    
    @property
    def weight(self):
        return self.linear.weight if self.linear is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        
        if not self._enabled:
            if self.bias_orig is not None:
                out = out + self.bias_orig
            return out
        
        if self._retain_bias and self.bias_orig is not None:
            out = out + self.bias_orig
        elif self.bias is not None:
            b = x.shape[0] // 2
            if b <= self.half_batch:
                # 使用 narrow 避免拷贝
                bias_first_half = self.bias.narrow(0, 0, b)
                bias_second_half = self.bias.narrow(0, self.half_batch, b)
            else:
                repeats = (b + self.half_batch - 1) // self.half_batch
                ndim = len(self.bias.shape) - 1
                bias_first_half = self.bias[:self.half_batch].repeat(repeats, *([1] * ndim))[:b]
                bias_second_half = self.bias[self.half_batch:self.batch_size].repeat(repeats, *([1] * ndim))[:b]
            
            bias_to_use = torch.cat([bias_first_half, bias_second_half], dim=0)
            out = out + bias_to_use
        
        return out


def wrap_affine_layers(module: nn.Module, name: str, batch_size: int, parent: Optional[nn.Module]) -> None:
    """递归包装所有仿射层"""
    if is_affine_layer(module):
        if parent is not None:
            parent._modules[name] = AffineLayerWrapper(module, batch_size)
    
    for child_name, child in module.named_children():
        if isinstance(module, AffineLayerWrapper) and child_name == 'linear':
            continue
        wrap_affine_layers(child, child_name, batch_size, module)


class ModelWrapper:
    """模型封装器：优化版本"""
    
    SUPPORTED_ACTIVATIONS = (nn.ReLU,)
    
    def __init__(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        batch_size: int = 128,
        device: str = "cpu"
    ):
        self.device = device
        self.batch_size = batch_size
        self._input_shape = input_shape
        
        self._model = deepcopy(model)
        self._model.eval()
        self._model.requires_grad_(False)
        self._model.to(device)
        
        self._model_dtype = next(self._model.parameters()).dtype
        
        wrap_affine_layers(self._model, 'model', batch_size, None)
        
        self._relu_modules = []
        self._affine_wrappers = []
        self._num_relus = 0
        self._find_and_register_relus()
        self._find_affine_wrappers()
        
        self._model_handles = None
        
        self._lambdas_to_cross = None
        self._patterns_buffer = []
        self._hooks_active = False
        
        # 预创建常量张量避免重复创建
        self._epsilon = torch.tensor([EPSILON], dtype=torch.float64, device=device)
        self._inf = torch.tensor([np.inf], dtype=torch.float64, device=device)
        self._one = torch.ones((1,), dtype=torch.float64, device=device)
        self._machine_eps = torch. tensor([torch.finfo(torch.float64).eps], dtype=torch. float64, device=device)
        
        self._retain_bias = False
        
        print("[ModelWrapper] Found %d ReLU layers" % self._num_relus)
    
    @property
    def input_shape(self):
        return self._input_shape
    
    @property
    def num_relus(self):
        return self._num_relus
    
    @property
    def model(self):
        return self._model
    
    def _find_and_register_relus(self):
        relu_order = []
        
        def order_hook(module, inp, out):
            relu_order.append(module)
        
        temp_hooks = []
        for module in self._model.modules():
            if isinstance(module, self.SUPPORTED_ACTIVATIONS):
                temp_hooks.append(module.register_forward_hook(order_hook))
        
        dummy = torch.ones((2,) + self._input_shape, device=self.device, dtype=self._model_dtype)
        with torch.no_grad():
            self._model(dummy)
        
        for h in temp_hooks:
            h.remove()
        
        self._relu_modules = relu_order
        
        for idx, module in enumerate(self._relu_modules):
            module.register_buffer("index", torch.tensor(idx), persistent=False)
        
        self._num_relus = len(self._relu_modules)
    
    def _find_affine_wrappers(self):
        for module in self._model.modules():
            if isinstance(module, AffineLayerWrapper):
                self._affine_wrappers.append(module)
    
    def _set_affine_wrappers_enabled(self, enabled: bool):
        for wrapper in self._affine_wrappers:
            wrapper.set_enabled(enabled)
    
    def _reset_state(self, batch_size: int):
        self._lambdas_to_cross = torch.full(
            (batch_size, self._num_relus),
            fill_value=np.inf,
            dtype=torch.float64,
            device=self.device
        )
        self._patterns_buffer = []
    
    def _attach_hooks(self):
        if self._model_handles is not None:
            return
        
        self._model_handles = []
        
        for module in self._relu_modules:
            handle = module.register_forward_hook(self._relu_hook)
            self._model_handles.append(handle)
        
        for module in self._affine_wrappers:
            pre_handle = module.register_forward_pre_hook(self._affine_pre_hook)
            post_handle = module.register_forward_hook(self._affine_post_hook)
            self._model_handles.extend([pre_handle, post_handle])
    
    def _remove_hooks(self):
        if self._model_handles is None:
            return
        for handle in self._model_handles:
            handle.remove()
        self._model_handles = None
    
    def _affine_pre_hook(self, module, inputs):
        module.retain_bias_(self._retain_bias)
        return inputs
    
    def _affine_post_hook(self, module, inputs, output):
        module.retain_bias_(False)
        return output
    
    def _relu_hook(self, module, inputs, output):
        if not self._hooks_active:
            return output
        
        relu_idx = module.index.item()
        pre_act = inputs[0]
        
        half_batch = pre_act.shape[0] // 2
        if half_batch == 0:
            return output
        
        # 使用 narrow 避免拷贝
        pre_act_data = pre_act.narrow(0, 0, half_batch)
        pre_act_dir = pre_act.narrow(0, half_batch, half_batch)
        
        # 激活模式 - 直接使用，不需要 clone（后续不修改）
        act_pattern = pre_act_data > 0
        self._patterns_buffer.append(act_pattern)
        
        # 使用 JIT 编译的 lambda 计算
        with torch.no_grad():
            min_lambda = _compute_lambdas_jit(
                pre_act_data, pre_act_dir,
                self._epsilon, self._inf, self._machine_eps
            )
            self._lambdas_to_cross[:, relu_idx] = min_lambda
        
        # 使用 in-place 操作修改方向部分的输出
        output[half_batch:] = pre_act_dir * act_pattern.to(pre_act_dir.dtype)
        
        return output
    
    def _forward(self, x: torch.Tensor, retain_bias: bool = False):
        self._retain_bias = retain_bias
        batch_size = x.shape[0] // 2
        self._reset_state(batch_size)
        self._hooks_active = True
        
        output = self._model(x)
        
        self._retain_bias = False
        self._hooks_active = False
        return output[:batch_size], output[batch_size:]
    
    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        """简单前向传播（不追踪）"""
        self._set_affine_wrappers_enabled(False)
        self._hooks_active = False
        
        x = x.to(device=self.device, dtype=self._model_dtype)
        with torch.no_grad():
            output = self._model(x)
        
        self._set_affine_wrappers_enabled(True)
        return output
    
    def get_region_state(self, x: torch.Tensor, direction: torch.Tensor, retain_bias: bool = False) -> RegionState:
        """获取区域状态"""
        self._attach_hooks()
        self._set_affine_wrappers_enabled(True)
        
        x = x.to(device=self.device, dtype=self._model_dtype)
        direction = direction.to(device=self.device, dtype=self._model_dtype)
        
        combined = torch.cat([x, direction], dim=0)
        
        with torch.no_grad():
            logits, _ = self._forward(combined, retain_bias=retain_bias)
        
        lambda_to_boundary, boundary_layer_idx = self._lambdas_to_cross.min(dim=1)
        
        return RegionState(
            activation_pattern=ActivationPattern(patterns=[p.clone() for p in self._patterns_buffer]),
            lambdas_per_layer=self._lambdas_to_cross.clone(),
            lambda_to_boundary=lambda_to_boundary,
            boundary_layer_idx=boundary_layer_idx,
            logits=logits.clone()
        )
    
    def get_activation_pattern_only(self, x: torch.Tensor) -> ActivationPattern:
        """只获取激活模式"""
        self._attach_hooks()
        self._set_affine_wrappers_enabled(True)
        
        x = x.to(device=self.device, dtype=self._model_dtype)
        
        zero_dir = torch.zeros_like(x)
        combined = torch.cat([x, zero_dir], dim=0)
        
        with torch.no_grad():
            self._forward(combined)
        
        return ActivationPattern(patterns=[p.clone() for p in self._patterns_buffer])
    
    def cleanup(self):
        self._remove_hooks()
        self._lambdas_to_cross = None
        self._patterns_buffer = []