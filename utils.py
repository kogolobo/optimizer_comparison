from dataclasses import dataclass, field
from typing import List
from trainer import Trainer
import matplotlib.pyplot as plt
import torch

@dataclass
class Experiment:
    optimizer_name: str
    optimizer_cls: type
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    eval_iterations: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)

    def add_trainer_state(self, trainer: Trainer) -> None:
        self.train_losses = trainer.train_losses
        self.eval_iterations = trainer.eval_iterations
        self.eval_losses = trainer.eval_losses
        self.accuracies = trainer.accuracies

def plot_train_losses(experiments: List[Experiment]) -> None:
    # Plot losses in different colors on a single plot, with a legend
    plt.figure(figsize=(12, 6))
    for exp in experiments:
        plt.plot(exp.train_losses, label=exp.optimizer_name, alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.show()

def plot_eval_losses(experiments: List[Experiment]) -> None:
    # Plot losses in different colors on a single plot, with a legend
    plt.figure(figsize=(12, 6))
    for exp in experiments:
        plt.plot(exp.eval_iterations, exp.eval_losses, label=exp.optimizer_name, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss Over Iterations')
    plt.legend()
    plt.show()

def plot_accuracies(experiments: List[Experiment]) -> None:
    # Plot losses in different colors on a single plot, with a legend
    plt.figure(figsize=(12, 6))
    for exp in experiments:
        plt.plot(exp.eval_iterations, exp.accuracies, label=exp.optimizer_name, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Iterations')
    plt.legend()
    plt.show()
