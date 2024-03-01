from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix
import argparse

from model import DistillBERT
from data import CIFAR10Dataset, SST2Dataset
from trainer import Trainer
from utils import Experiment, plot_train_losses, plot_eval_losses, plot_accuracies
from transformers.trainer_utils import set_seed
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='distillbert')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
set_seed(args.seed)

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
        dataset = CIFAR10Dataset(args.train_batch_size, args.eval_batch_size)
    elif args.model == 'distillbert':
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        dataset = SST2Dataset(args.train_batch_size, args.eval_batch_size, tokenizer)

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, dataset, criterion, exp.optimizer_cls, args.lr)
    trainer.train(args.epochs)
    exp.add_trainer_state(trainer)

plot_train_losses(experiments)
plot_eval_losses(experiments)
plot_accuracies(experiments)