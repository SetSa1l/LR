"""
run_lc_mnist.py

复现实验（Patel & Montúfar 2025）用于 MNIST MLP（Figure 3 风格）。
仅在1000样本上训练，按指数形式保存checkpoint，不计算LC。
支持从指定epoch继续训练。
保存：model_epoch{指数}.pth
"""

import os
import random
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# -------------------------
# 参数（新增--resume参数）
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:2' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--subset_size', type=int, default=1000, help='固定使用1000张MNIST子集训练')
parser.add_argument('--batch_size', type=int, default=None, help='如果None使用全量子集作为一个batch（默认）')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=600, help='训练总轮次')
parser.add_argument('--save_dir', type=str, default='./saved/1000')
parser.add_argument('--sigma', type=float, default=0.05, help='保留参数（原LC计算用，当前未使用）')
parser.add_argument('--resume', type=str, default=None, help='从指定checkpoint继续训练（例如：./saved/1000/model_epoch100.pth）')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# -------------------------
# 固定随机种子
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

device = torch.device(args.device)

# -------------------------
# 简单MLP（4 hidden layers × 200），与论文一致
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=200, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

        # 初始化：He初始化并把权重放大2x（论文中Figure3使用2x He）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.weight.data.mul_(2.0)  # 2x He init
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.out(x)
        return x

# -------------------------
# 载入MNIST子集（固定1000张，十类各100张）
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)

# 抽取十个类上的各100张样本，共1000张
class_indices = {i: [] for i in range(10)}
for idx, (img, label) in enumerate(full_train):
    class_indices[label].append(idx)
selected_indices = []
per_class = args.subset_size // 10
for i in range(10):
    selected_indices.extend(class_indices[i][:per_class])
train_subset = Subset(full_train, selected_indices)
batch_size = args.batch_size or args.subset_size  # 默认一次forward用全部子集
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

# -------------------------
# 训练主循环（支持从断点继续训练）
# -------------------------
def train_and_save():
    # 初始化模型、损失函数、优化器
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 初始化训练起点（默认从0开始）
    start_epoch = 0

    # 从checkpoint恢复（如果指定）
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"指定的checkpoint不存在：{args.resume}")
        
        # 加载checkpoint
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # 恢复模型权重
        model.load_state_dict(checkpoint['state_dict'])
        # 恢复优化器状态（保证学习率、动量等续接）
        opt.load_state_dict(checkpoint['optimizer'])
        # 恢复训练进度（从下一个epoch开始）
        start_epoch = checkpoint['epoch']
        print(f"已从 checkpoint 恢复训练：{args.resume}")
        print(f"将从 epoch {start_epoch + 1} 开始训练（原已训练到 epoch {start_epoch}）")

    exp_epochs = []
    current = 1
    # 先收集指数间隔的epoch
    while current <= args.epochs:
        exp_epochs.append(current)
        current *= 2

    # 再添加每10000 epoch的强制保存点（如果不在指数间隔中）
    # 计算10000的整数倍中 <= args.epochs的所有值
    max_10000 = (args.epochs // 10000) * 10000
    for epoch in range(10000, max_10000 + 1, 10000):
        if epoch not in exp_epochs:  # 避免重复添加
            exp_epochs.append(epoch)

    # 最后按升序排序
    exp_epochs = sorted(exp_epochs)



    print(f"训练总轮次：{args.epochs}")
    print(f"指数间隔保存checkpoint的epoch：{exp_epochs}")

    # 从start_epoch开始训练（注意：range是左闭右开，所以终点是args.epochs）
    for ep in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # 计算平均损失（当前epoch是ep+1，因为ep从0开始计数）
        current_epoch = ep + 1
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {current_epoch}/{args.epochs}: avg_loss={avg_loss:.4e}")

        # 按指数间隔保存checkpoint（即使是续训，也会保存符合指数间隔的epoch）
        if current_epoch in exp_epochs:
            checkpoint_path = os.path.join(args.save_dir, f'model_epoch{current_epoch}.pth')
            torch.save({
                'epoch': current_epoch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'loss': avg_loss,
                'config': vars(args)
            }, checkpoint_path)
            print(f"已保存checkpoint到：{checkpoint_path}")
        #保存最后一个epoch的checkpoint
        if current_epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f'model_epoch{current_epoch}.pth')
            torch.save({
                'epoch': current_epoch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'loss': avg_loss,
                'config': vars(args)
            }, checkpoint_path)
            print(f"已保存最终checkpoint到：{checkpoint_path}")
    print("训练完成！所有checkpoint已保存到：", args.save_dir)

if __name__ == '__main__':
    train_and_save()