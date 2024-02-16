import argparse
from torch import nn, optim
from sklearn.metrics import accuracy_score, confusion_matrix

from model import FeedforwardNeuralNet
from data import MNISTDataset, CIFAR10Dataset
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=512)
args = parser.parse_args()

dataset = MNISTDataset(args.train_batch_size, args.eval_batch_size)
model = FeedforwardNeuralNet(dataset.input_size, dataset.output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

trainer = Trainer(model, dataset, criterion, optimizer)
trainer.train(args.epochs)
predictions, true_labels = trainer.evaluate()

accuracy = accuracy_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)

print("Accuracy =", accuracy)
print("Confusion Matrix =\n", conf_matrix)