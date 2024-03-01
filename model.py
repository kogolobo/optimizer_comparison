from torch import nn

class FeedforwardNeuralNet(nn.Module):
    HIDDEN_SIZES = [128, 64]

    def __init__(self, input_size, num_classes):
        super(FeedforwardNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, self.HIDDEN_SIZES[0]) 
        self.fc2 = nn.Linear(self.HIDDEN_SIZES[0], self.HIDDEN_SIZES[1])
        self.fc3 = nn.Linear(self.HIDDEN_SIZES[1], num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
