import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
from torch.autograd import grad

import math
import torch
import torch.nn as nn

def compute_local_complexity(model, batch_x, activations, sigma=0.05, n_samples=1):
    """
    根据论文近似计算 Local Complexity (LC).
    输入:
        model: PyTorch 模型
        batch_x: 输入 batch (tensor), 已经在 device 上
        activations: forward hook 收集的 pre-activation 输出
        sigma: 高斯噪声标准差
        n_samples: 蒙特卡洛采样次数 (1 已足够, >1 用于平滑曲线)
    返回:
        LC 标量
    """

    model.eval()
    LC_sum = 0.0
    B = batch_x.shape[0]
    x = batch_x.requires_grad_(True)

    # 保存原始 bias
    original_biases = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            original_biases.append(m.bias.data.clone())

    for _ in range(n_samples):

        # 添加 bias 噪声
        idx = 0
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                noise = torch.randn_like(m.bias) * sigma
                with torch.no_grad():
                    m.bias.add_(noise)
                idx += 1

        # 重新 forward 得到带噪声的 pre-activation
        activations.clear()
        _ = model(x)

        # 重新计算梯度
        from torch.autograd import grad
        for layer_name, z in activations.items():

            if z.dim() == 4:
                B2, C, H, W = z.shape
                z_flat = z.reshape(B2, C * H * W)
            else:
                z_flat = z

            # 高斯密度 ρ(b - z(x))   ~ exp(-(z/sigma)^2 /2)
            rho = torch.exp(- (z_flat / sigma)**2 / 2) / (sigma * math.sqrt(2*math.pi))  # (B,N)

            B2, N = z_flat.shape
            for neuron_idx in range(N):
                scalar = z_flat[:, neuron_idx].sum()

                g = torch.autograd.grad(
                    scalar, x,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if g is None:
                    continue
                        
                grad_norm = g.view(B2, -1).norm(dim=1)      # (B,)
                rho_i = rho[:, neuron_idx]                  # (B,)

                LC_sum += (grad_norm * rho_i).sum().item()

        # 恢复 bias
        idx = 0
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                with torch.no_grad():
                    m.bias.data.copy_(original_biases[idx])
                idx += 1

    model.train()
    return LC_sum / (B * n_samples)


def compute_local_complexity_simple(model, batch_x, activations):

    model.eval()
    LC_sum = 0.0
    neuron_grads, _ = compute_neuron_gradients(activations, batch_x)
    
    for grad_norm in neuron_grads.values():
        #一个grad_norm代表一个神经元在batch_size个样本上的梯度值
        LC_sum +=(grad_norm).sum().item()
    
    model.train()
    return LC_sum/len(batch_x)

def compute_neuron_gradients(activations,  x):
    # 确保可导
    x = x.requires_grad_(True)

    # 清空旧结果
    neuron_grads={}
    neuron_values={}
    for layer_name, z in activations.items():
    # --- 关键：卷积层处理，统一为 (B, N) ---
        if z.dim() == 4:  # conv layer pre-activation
            B, C, H, W = z.shape
            z_flat = z.reshape(B, C * H * W)
        elif z.dim() == 2:  # linear layer pre-activation
            z_flat = z
            B, _ = z_flat.shape
        else:
            raise ValueError(f"Unsupported activation shape {z.shape}")

        B, N = z_flat.shape

        for neuron_idx in range(N):
            scalar = z_flat[:, neuron_idx].sum()
            grad = torch.autograd.grad(
                scalar, x,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            # 有些神经元可能不影响输入，跳过
            if grad is None:
                continue
            # CNN 输入是 (B, C, H, W)，梯度范数应在空间维度上求
            grad_norm = grad.view(B, -1).norm(dim=1)  # (B,)

            neuron_grads[(layer_name, neuron_idx)] = grad_norm
            neuron_values[(layer_name, neuron_idx)] = z_flat[:, neuron_idx].detach()

    return neuron_grads, neuron_values