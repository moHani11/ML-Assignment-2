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
    
    

inpt = torch.randn(3,3)
flatten = nn.Flatten()
# flattened_inpt = torch.flatten(inpt)
# print(inpt)
# print(flattened_inpt)
# linearLayer = nn.Linear(3*3,2)
# out = linearLayer(flattened_inpt)
# print(out)
# print(f"\n Weights: {linearLayer.weight} \n Bias: {linearLayer.bias} ")

print(inpt.device)
inpt2 = torch.randn(9,1)
# print(inpt2)

net = customNet()
out = net.predict(inpt2)

print(out)