import torch
from torch import nn, optim
from typing import Any

from data import Dataset

class Trainer:
    def __init__(self, 
        model: nn.Module, 
        dataset: Dataset, 
        criterion: Any, 
        optimizer: optim.Optimizer
    ) -> None:
        
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs: int):
        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in self.dataset.train_loader:
                inputs = inputs.view(inputs.shape[0], -1)

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            else:
                print(f"Epoch {epoch} Training loss: {running_loss/len(self.dataset.train_loader)}")

    def evaluate(self):
        predictions, true_labels = [], []
        for inputs, labels in self.dataset.test_loader:
            inputs = inputs.view(inputs.shape[0], -1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            with torch.no_grad():
                logps = self.model(inputs)

            _, predicted = torch.max(logps, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())

        return predictions, true_labels