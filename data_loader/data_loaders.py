from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
import numpy as np

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
   
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):    
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistSubsetDataLoader(BaseDataLoader):
    """
    从MNIST数据集中每个类别选取100个样本的子数据集加载器
    总样本数：10类 × 100 = 1000个
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 数据预处理与原始MNIST保持一致
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.data_dir = data_dir
        # 加载完整MNIST数据集
        full_dataset = datasets.MNIST(
            self.data_dir, 
            train=training, 
            download=True, 
            transform=trsfm
        )
        
        # 筛选每个类别的100个样本
        self.subset_dataset = self._create_balanced_subset(full_dataset, num_samples_per_class=100)
        
        # 调用父类构造函数初始化数据加载器
        super().__init__(
            self.subset_dataset, 
            batch_size, 
            shuffle, 
            validation_split, 
            num_workers
        )
    
    def _create_balanced_subset(self, full_dataset, num_samples_per_class=100):
        """创建每个类别包含指定数量样本的平衡子数据集"""
        # 获取所有样本的标签
        if isinstance(full_dataset.targets, torch.Tensor):
            targets = full_dataset.targets.numpy()
        else:
            targets = np.array(full_dataset.targets)
        
        # 收集每个类别的索引
        class_indices = {}
        for class_label in range(10):  # MNIST有10个类别（0-9）
            indices = np.where(targets == class_label)[0]
            # 随机选择指定数量的样本（确保不超过该类别的总样本数）
            selected_indices = np.random.choice(
                indices, 
                size=min(num_samples_per_class, len(indices)),
                replace=False  # 不重复选择
            )
            class_indices[class_label] = selected_indices
        
        # 合并所有类别的选中索引
        all_selected_indices = np.concatenate(list(class_indices.values()))
        # 打乱索引顺序（避免同类样本集中）
        np.random.shuffle(all_selected_indices)
        
        # 创建子数据集
        return torch.utils.data.Subset(full_dataset, all_selected_indices)
    
    
class Cifar10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)