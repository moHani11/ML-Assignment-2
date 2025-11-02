import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F 
from collections import OrderedDict
from sklearn.utils import shuffle

class customNet(nn.Module):
    
    def __init__(self, input_size, num_of_classes, learning_rate):
        super(customNet, self).__init__()
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        self.model = nn.Sequential(
        OrderedDict(
                [
                ("Linear Layer #1", nn.Linear(input_size, 20)),
                ("relu1", nn.ReLU()),
                ("Linear Layer #2", nn.Linear(20, num_of_classes)),
                 # ("relu", nn.ReLU()),
                ]
            )
        )
    
    def predict(self, x):
        x = torch.flatten(x)
        return self.model(x)
        
    
    def train(self, X, Y, batch_size = 64, epochs = 1):
        X, Y = shuffle(X, Y)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            for sample_idx in range(batch_size):
                i = epoch*batch_size + sample_idx
                y_predicted = self.predict(X[i])
                loss = self.loss_function(Y, y_predicted)
                loss.backward()
            self.optimizer.step()


