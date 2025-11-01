import torch
from torch.autograd import Variable
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim


# print(torch.cuda.is_available())
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        flatten = nn.Flatten()
        self.fc = nn.Linear(1,1)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    def predict(self, x):
        return self.forward(x)

net = Net()
net.cuda()
# print(net)    


# input = Variable(torch.randn(1,1,1), requires_grad=True)
# print(input)

# print(net(input))

def criterion(out, label):
    return (label - out)**2
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

for epoch in range(10):
    for i, data2 in enumerate(data):
        X, Y = iter(data2)
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([X]), requires_grad=True)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))