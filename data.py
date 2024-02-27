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

class data_load(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode(text)
        return torch.tensor(encoding.ids)
class data_handle:
    def __init__(self, train_data, test_data, tokenizer, batch_size=1, shuffle_train=True, shuffle_test=False):
        self.train_dataset = data_load(train_data, tokenizer)
        self.test_dataset = data_load(test_data, tokenizer)
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train
            )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test
            )
    def get_train_loader(self):
        return self.train_loader
    def get_test_loader(self):
        return self.test_loader
    def numericalize(text):
        return [vocab[token] for token in tokenizer(text)]

