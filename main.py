from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix
import argparse

from model import FeedforwardNeuralNet
from data import MNISTDataset, CIFAR10Dataset
from trainer import Trainer
from utils import Experiment, plot_train_losses, plot_eval_losses, plot_accuracies


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--model',type=str,default='resnet')
args = parser.parse_args()

print("Running with the following arguments: ", args)

experiments = [
    Experiment("SGD", optim.SGD),
    Experiment("Adagrad", optim.Adagrad),
    Experiment("RMSprop", optim.RMSprop),
    Experiment("Adam", optim.Adam)
]

for exp in experiments:
    print(f"Training with optimizer: {exp.optimizer_name}")

    if args.model == 'resnet':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

    criterion = nn.CrossEntropyLoss()
    dataset = CIFAR10Dataset(args.train_batch_size, args.eval_batch_size)
    trainer = Trainer(model, dataset, criterion, exp.optimizer_cls, args.lr)
    trainer.train(args.epochs)
    exp.add_trainer_state(trainer)

plot_train_losses(experiments)
plot_eval_losses(experiments)
plot_accuracies(experiments)