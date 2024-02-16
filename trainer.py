import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from typing import Tuple, List, Any

class Trainer:
    def __init__(self, model: nn.Module, dataset: Dataset, criterion: Any, optimizer: optim.Optimizer) -> None:
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer

        self.all_train_losses: List[float] = []
        self.eval_loss_history: List[Tuple[int, float]] = []  # (iteration, eval_loss)
        self.accuracies: List[float] = []
        self.predictions: List[int] = []
        self.true_labels: List[int] = []

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs: int, eval_every_n_iterations: int = 100) -> None:
        iteration_count = 0

        for epoch in range(epochs):
            self.model.train()
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

                self.all_train_losses.append(loss.item())
                iteration_count += 1

                if iteration_count % eval_every_n_iterations == 0:
                    eval_loss, accuracy = self.evaluate()
                    self.eval_loss_history.append((iteration_count, eval_loss))
                    self.accuracies.append(accuracy)
                    print(f"Iteration {iteration_count}, Training loss: {loss.item():.4f}, Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

    
        # Evaluate at the end of training
        eval_loss, accuracy = self.evaluate()
        self.eval_loss_history.append((iteration_count, eval_loss))
        self.accuracies.append(accuracy)
        print(f"Iteration {iteration_count}, Training loss: {loss.item():.4f}, Eval loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")


        self.plot_losses()
        self.plot_accuracy()

    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        self.predictions = []  # Reset predictions
        self.true_labels = []  # Reset true labels

        with torch.no_grad():
            for inputs, labels in self.dataset.test_loader:
                inputs = inputs.view(inputs.shape[0], -1)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                output = self.model(inputs)
                loss = self.criterion(output, labels)
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                self.predictions.extend(predicted.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())

        average_loss = total_loss / len(self.dataset.test_loader)
        accuracy = accuracy_score(self.true_labels, self.predictions)
        return average_loss, accuracy

    def plot_losses(self) -> None:
        plt.figure(figsize=(12, 6))
        eval_iterations, eval_losses = zip(*self.eval_loss_history)
        plt.plot(self.all_train_losses, label='Training Loss', alpha=0.5)
        plt.plot(eval_iterations, eval_losses, label='Evaluation Loss', marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Iterations')
        plt.legend()
        plt.show()

    def plot_accuracy(self) -> None:
        if self.accuracies:  # Ensure there are accuracy data points to plot
            eval_iterations, _ = zip(*self.eval_loss_history)
            plt.figure(figsize=(12, 6))
            plt.plot(eval_iterations, self.accuracies, label='Accuracy', marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy Over Iterations')
            plt.legend()
            plt.show()
