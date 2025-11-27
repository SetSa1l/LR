#正常训练一个MLP模型，在全部mnist数据集上训练，对数间隔保存checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=200, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.out(x)
        return x

# 定义数据集和数据加载器
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 定义模型、优化器和损失函数
model = MLP().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 创建保存目录
save_dir = Path('saved/basic_mlp')
save_dir.mkdir(parents=True, exist_ok=True)

# 训练模型
iter_per_epoch = len(train_loader)
num_steps = 500000
batch_size = 128
exp_iterations = []
current = 1

while current <= num_steps:
    exp_iterations.append(current)
    current *= 2

#按照iteration训练,每个iteration训练一个batch
train_step = 0

while True:

    if train_step >= num_steps:
        break

    total_loss = 0.0
    for ims, labs in tqdm(train_loader, desc=f"train_step:{train_step}-{train_step+iter_per_epoch}"):
        ims, labs = ims.to('cuda'), labs.to('cuda')
        ims = ims.view(ims.size(0), -1)
        optimizer.zero_grad()
        output = model(ims)

        loss = criterion(output, labs)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_step += 1

        if train_step in exp_iterations:
            save_path = save_dir / f'step{train_step}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Saved checkpoint at step {train_step}')

    
    #输出当前平均损失
    avg_loss = total_loss / iter_per_epoch
    print(f'train_step:{train_step}, avg_loss:{avg_loss}')
        
        