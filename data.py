import torch
from torchvision import datasets, transforms
from dataclasses import dataclass

@dataclass
class Dataset:
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    input_size: int
    output_size: int


class MNISTDataset(Dataset):
    input_size: int = 784
    output_size: int = 10

    def __init__(self, trian_batch_size: int = 64, eval_batch_size: int = 64):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=trian_batch_size, shuffle=True)
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=True)

class CIFAR10Dataset(Dataset):
    input_size: int = 32 * 32 * 3
    output_size: int = 10

    def __init__(self, trian_batch_size: int = 64, eval_batch_size: int = 64):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=trian_batch_size, shuffle=True)
        testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=True)