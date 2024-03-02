import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score

from typing import Dict, Tuple, List, Any
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding

from data import Dataset

class Trainer:
    def __init__(
            self, 
            model: nn.Module, 
            dataset: Dataset, 
            criterion: Any, 
            optimizer_cls: type, 
            lr: float
        ) -> None:

        self.model = model
        self.dataset = dataset
        self.criterion = criterion

        self.train_losses: List[float] = []
        self.eval_iterations: List[int] = []
        self.eval_losses: List[float] = []
        self.accuracies: List[float] = []
        self.predictions: List[int] = []
        self.true_labels: List[int] = []

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optimizer_cls(model.parameters(), lr=lr)

    def train(self, epochs: int, eval_every_n_iterations: int = 100) -> None:
        iteration_count = 0

        for epoch in range(epochs):
            self.model.train()
            for batch in self.dataset.train_loader:
                inputs, labels = self.parse_batch(batch)

                self.optimizer.zero_grad()

                if isinstance(inputs, dict):
                    output = self.model(**inputs)[0]
                else:
                    output = self.model(inputs)

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
            
                self.train_losses.append(loss.item())
                iteration_count += 1

                if iteration_count % eval_every_n_iterations == 0:
                    eval_loss, accuracy = self.evaluate()
                    self.eval_iterations.append(iteration_count)
                    self.eval_losses.append(eval_loss)
                    self.accuracies.append(accuracy)
                    print(f"Iteration {iteration_count}, Training loss: {loss.item():.4f}, " + 
                          f"Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    
        # Evaluate at the end of training
        eval_loss, accuracy = self.evaluate()
        self.eval_iterations.append(iteration_count)
        self.eval_losses.append(eval_loss)
        self.accuracies.append(accuracy)
        print(f"Iteration {iteration_count}, Training loss: {loss.item():.4f}, " +
              f"Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        self.predictions = []  # Reset predictions
        self.true_labels = []  # Reset true labels

        with torch.no_grad():
            for batch in self.dataset.test_loader:
                inputs, labels = self.parse_batch(batch)

                if isinstance(inputs, dict):
                    output = self.model(**inputs)[0]
                else:
                    output = self.model(inputs)
                loss = self.criterion(output, labels)
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                self.predictions.extend(predicted.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())

        average_loss = total_loss / len(self.dataset.test_loader)
        accuracy = accuracy_score(self.true_labels, self.predictions)
        return average_loss, accuracy

    def parse_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor | Dict[str, Tensor], Tensor]:
        if isinstance(batch, BatchEncoding):
            inputs = {key: batch[key] for key in batch if key != "labels"}
            labels = batch["labels"]

            if torch.cuda.is_available():
                for key in inputs:
                    inputs[key] = inputs[key].cuda()
                labels = labels.cuda()

        else:
            inputs, labels = batch
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

        return inputs, labels

    def __del__(self) -> None:
        if torch.cuda is not None and torch.cuda.is_available():
            self.model.cpu()
