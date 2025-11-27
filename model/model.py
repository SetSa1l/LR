import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
class MLP(BaseModel):
    def __init__(self, input_size=28*28, hidden=200, num_classed=10, scale_weight=-1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classed)
        self.relu = nn.ReLU()
        self.scale_weight = scale_weight

        # if scale_weight > 0:
        #     for m in self.modules():
        #         if isinstance(m, nn.Linear):
        #             m.weight.data.mul_(scale_weight)
        #             if m.bias is not None:
        #                 m.bias.data.mul_(scale_weight)
        
        # # 初始化：He初始化并把权重放大2x（论文中Figure3使用2x He）
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #         m.weight.data.mul_(2.0)  # 2x He init
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(self, x):
        # 自适应展平输入：无论输入是什么形状，都调整为 (batch_size, input_size)
        original_shape = x.shape
        total_elements = x.numel()
        expected_features = self.fc1.in_features
        
        # 如果输入维度大于2，先展平
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 检查特征维度是否匹配
        if x.size(1) != expected_features:
            # 如果特征维度不匹配，尝试根据总元素数推断正确的batch_size
            # 计算可能的batch_size：total_elements / expected_features
            inferred_batch_size = total_elements // expected_features
            if inferred_batch_size > 0 and (inferred_batch_size * expected_features) == total_elements:
                # 如果能够整除，说明可以reshape为 (inferred_batch_size, expected_features)
                x = x.view(inferred_batch_size, expected_features)
            else:
                # 如果无法整除，尝试保持当前batch_size，调整特征维度
                current_batch = x.size(0)
                current_features = x.size(1)
                if current_features < expected_features:
                    # 特征不足，用零填充
                    padding = torch.zeros(current_batch, expected_features - current_features, 
                                         device=x.device, dtype=x.dtype)
                    x = torch.cat([x, padding], dim=1)
                elif current_features > expected_features:
                    # 特征过多，截断
                    x = x[:, :expected_features]
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.out(x)
        return x

class CNN(BaseModel):
    def __init__(self, input_channel=3, num_classed=10, scale_weight=-1):
        super().__init__()
        # 5个卷积层（文中实验默认CNN架构为5个卷积层+2个线性层，无BatchNorm）
        self.conv_layers = nn.Sequential(
            # 卷积层1
            nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # 下采样，保持与原结构池化方式一致
            
            # 卷积层2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            
            # 卷积层3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            
            # 卷积层4
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            
            # 卷积层5
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        
        # 自适应池化（确保输入到线性层的特征维度一致）
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2个线性层（文中实验默认配置）
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 32),  # 第一层线性层，维度与卷积输出匹配
            nn.ReLU(),
            nn.Linear(32, num_classed)  # 输出层，对应CIFAR10的10个类别
        )

        self.scale_weight = scale_weight

        if scale_weight > 0:
        # 放大权重,采用He初始化，并把权重放大scale_weight倍
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.weight.data.mul_(scale_weight)
                    if m.bias is not None:
                        m.bias.data.mul_(scale_weight)
                        nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc_layers(x)
        return x



class BasicBlock(nn.Module):
    '''Pre-activation BasicBlock for ResNet18'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, **kwargs):
        super(BasicBlock, self).__init__()
        # Pre-activation: BN -> ReLU -> Conv
        if bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        else:
            self.bn1 = nn.Identity()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if bn:
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # Pre-activation: BN -> ReLU -> Conv
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet18(BaseModel):
    def __init__(self, input_channel=3, num_classed=10, BN=False, width=16):
        super().__init__()
        # Pre-activation ResNet18 for CIFAR10 with width=16
        self.in_planes = width
        c = width
        
        # 第一层：CIFAR10 配置 (kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        # 构建层：width 16 -> 32 -> 64 -> 128
        self.layer1 = self._make_layer(BasicBlock, c, 2, stride=1, bn=BN)
        self.layer2 = self._make_layer(BasicBlock, 2*c, 2, stride=2, bn=BN)
        self.layer3 = self._make_layer(BasicBlock, 4*c, 2, stride=2, bn=BN)
        self.layer4 = self._make_layer(BasicBlock, 8*c, 2, stride=2, bn=BN)
        
        # CIFAR10 使用 avg_pool2d(out, 4)
        self.fc = nn.Linear(8*c * BasicBlock.expansion, num_classed)
    
    def _make_layer(self, block, planes, num_blocks, stride, bn=True):
        # eg: [stride, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn=bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Pre-activation: 第一层直接 conv，没有 BN 和 ReLU
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # CIFAR10: 32x32 -> 经过 stride=2 的层后变为 4x4
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

