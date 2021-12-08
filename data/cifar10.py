from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import os


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data_raw/CIFAR10', batch_size=32):
        super().__init__()
        self.name = 'CIFAR10'
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        CIFAR10(root=self.data_dir, download=True, train=True)
        CIFAR10(root=self.data_dir, download=True, train=False)

    def setup(self, stage=None):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
        self.cifar10_train, self.cifar10_val = random_split(CIFAR10(root=self.data_dir, download=False, train=True, transform=self.transform_train), [45000, 5000])
        self.cifar10_test = CIFAR10(root=self.data_dir, download=False, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, pin_memory=True)