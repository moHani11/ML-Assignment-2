import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F 
from collections import OrderedDict
from sklearn.utils import shuffle
from dataLoaders import load_train_val, load_test


class customNet(nn.Module):
    
    def __init__(self, input_size, num_of_classes, learning_rate):
        super(customNet, self).__init__()

        self.num_of_classes = num_of_classes
        self.model = nn.Sequential(
        OrderedDict(
                [
                ("Hidden Layer #1", nn.Linear(input_size, input_size//3)),
                ("relu1", nn.ReLU()),
                ("Hidden Layer #2", nn.Linear(input_size//3, self.num_of_classes*3)),
                ("relu1", nn.ReLU()),
                ("Output Layer", nn.Linear(self.num_of_classes*3, self.num_of_classes)),
                 # ("relu", nn.ReLU()),
                ]
            )
        )
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)


    def predict(self, x):
        x = torch.flatten(x)
        return self.model(x)
        
    
    def train(self, X, Y, batch_size = 64, epochs = 1):
        X, Y = shuffle(X, Y)
        
        for epoch in range(epochs):
            loss = 0.0
            total_loss = 0.0

            self.optimizer.zero_grad()
            for sample_idx in range(batch_size):
                i = epoch*batch_size + sample_idx
                y_predicted = self.predict(X[i])
                # print(f"\n\n {y_predicted} \n\n")
                target = torch.zeros(size=(10,0))
                target = torch.zeros(self.num_of_classes, dtype=torch.float)
                target[Y[i]] = 1.0
                # print(f"\n\n {target} \n\n")

                loss = self.loss_function(target, y_predicted)
                print(f"Loss: {loss}")

                loss.backward()
                total_loss += loss
            self.optimizer.step()

            # i = epoch*batch_size + sample_idx
            # y_predicted = self.predict(X[epoch*batch_size: (epoch+1)*batch_size])
            # loss = self.loss_function(Y[epoch*batch_size: (epoch+1)*batch_size], y_predicted)
            print(f"Total Loss: {total_loss}")
            # loss.backward()
            # self.optimizer.step()




x_train, x_val, y_train, y_val = load_train_val(pytorch=True)

# print(x_train[0])    
print(f"number of training samples: {len(x_train)}")    
print(f"Input Images size: {x_train[0].shape}")
INPUT_DATA_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001


myNet = customNet(INPUT_DATA_SIZE, NUM_CLASSES, LEARNING_RATE)
batch_size = 32
epochs = 1
myNet.train(x_train, y_train, batch_size=batch_size, epochs=epochs)