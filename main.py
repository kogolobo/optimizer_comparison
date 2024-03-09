from torch import nn, optim
import argparse

from torchvision.models import resnet50, ResNet50_Weights
from transformers.trainer_utils import set_seed
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from data import CIFAR10Dataset, SST2Dataset, MNLIDataset
from trainer import Trainer
from utils import Experiment, plot_train_losses, plot_eval_losses, plot_accuracies

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='distillbert')
parser.add_argument('--dataset', type=str, default='sst2')
parser.add_argument('--eval_iterations', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

print("Running with the following arguments: ", args)

experiments = [
    Experiment("SGD", optim.SGD),
    Experiment("Adagrad", optim.Adagrad),
    Experiment("RMSprop", optim.RMSprop),
    Experiment("Adam", optim.Adam)
]

text_datasets = {
    "sst2": SST2Dataset,
    "mnli": MNLIDataset
}

vision_datasets = {
    "cifar10": CIFAR10Dataset
}

for exp in experiments:
    set_seed(args.seed)

    print(f"Training with optimizer: {exp.optimizer_name}")

    if args.model == 'resnet':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        dataset = vision_datasets[args.dataset](args.train_batch_size, args.eval_batch_size)
    elif args.model == 'distillbert':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        dataset = text_datasets[args.dataset](args.train_batch_size, args.eval_batch_size, tokenizer)
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=dataset.num_labels)
    else:
        raise ValueError(f"Model {args.model} not supported")

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model, 
        dataset, 
        criterion, 
        exp.optimizer_cls, 
        args.lr
    )
    trainer.train(args.epochs, args.eval_iterations)
    exp.add_trainer_state(trainer)

plot_train_losses(experiments)
plot_eval_losses(experiments)
plot_accuracies(experiments)
