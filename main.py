import argparse
from torch import nn, optim
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50, ResNet50_Weights
from transformers import DistilBertTokenizer, DistilBertModel

from data import MNISTDataset, CIFAR10Dataset
from trainer import Trainer_Image, Trainer_Text

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--model',type=str,default='resnet')
args = parser.parse_args()

print("Running with the following arguments: ", args)

if args.model == 'resnet':
    dataset = CIFAR10Dataset(args.train_batch_size, args.eval_batch_size)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    trainer = Trainer_Image(model, dataset, criterion, optimizer)
    trainer.train(args.epochs)
    trainer.evaluate()
    conf_matrix = confusion_matrix(trainer.true_labels, trainer.predictions)
    print("Confusion Matrix =\n", conf_matrix)
elif args.model == 'distillbert':
    dataset = None # TODO
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    trainer = Trainer_Image(model, dataset, criterion, optimizer)
    trainer.train(args.epochs)
    trainer.evaluate()
    conf_matrix = confusion_matrix(trainer.true_labels, trainer.predictions)
    print("Confusion Matrix =\n", conf_matrix)

    # text = "Hello, world!"
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
else:
    print('Input model not recognized :/')