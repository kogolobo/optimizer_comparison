from typing import Dict, Tuple
import torch
from torchvision import datasets, transforms
from dataclasses import dataclass
from torchtext.datasets import SST2
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding
from datasets import Dataset as HFDataset

@dataclass
class Dataset:
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

class CIFAR10Dataset(Dataset):
    # Source: https://github.com/fastai/imagenet-fast/blob/master/cifar10/train_cifar10.py
    transfomrs = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914 , 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159])
        ])

    def __init__(self, trian_batch_size: int = 64, eval_batch_size: int = 64):
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=self.transfomrs)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=trian_batch_size, shuffle=True)
        testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=self.transfomrs)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False)

class SST2Dataset(Dataset):
    def __init__(
            self, 
            train_batch_size: int = 64, 
            eval_batch_size: int = 64, 
            tokenizer: PreTrainedTokenizerFast = None
    ) -> None:
        train_dataset = [item for item in SST2(split='train')]
        train_sentences, train_labels = list(zip(*train_dataset))
        train_dataset = HFDataset.from_dict({'sentence': train_sentences, 'label': train_labels})

        test_dataset = [item for item in SST2(split='dev')]
        test_sentences, test_labels = list(zip(*test_dataset))
        test_dataset = HFDataset.from_dict({'sentence': test_sentences, 'label': test_labels})
        
        self.tokenizer = tokenizer
        train_dataset = train_dataset.map(self.tokenize)
        test_dataset = test_dataset.map(self.tokenize)

        # Remove columns that are not needed
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True, 
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )

    def tokenize(self, data):
        return self.tokenizer(data['sentence'], padding=False, truncation=True)