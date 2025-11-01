import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F 
from collections import OrderedDict

class customNet(nn.Module):
    
    def __init__(self):
        super(customNet, self).__init__()
        self.model = nn.Sequential(
        OrderedDict(
                [
                ("Linear Layer #1", nn.Linear(9, 20)),
                ("relu1", nn.ReLU()),
                ("Linear Layer #2", nn.Linear(20, 1)),
                # ("relu", nn.ReLU()),
                ]
            )
        )
    
    def predict(self, x):
        x = torch.flatten(x)
        return self.model(x)
    

from dataLoaders import load_train_val, load_test
x_train, x_val, y_train, y_val = load_train_val()


    
    