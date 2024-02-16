import argparse
from torch import nn, optim
from sklearn.metrics import confusion_matrix

from model import FeedforwardNeuralNet
from data import MNISTDataset, CIFAR10Dataset
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
args = parser.parse_args()

print("Running with the following arguments: ", args)

dataset = MNISTDataset(args.train_batch_size, args.eval_batch_size)
model = FeedforwardNeuralNet(dataset.input_size, dataset.output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

trainer = Trainer(model, dataset, criterion, optimizer)
trainer.train(args.epochs)
trainer.evaluate()
conf_matrix = confusion_matrix(trainer.true_labels, trainer.predictions)
print("Confusion Matrix =\n", conf_matrix)