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
                ("Hidden Layer #1", nn.Linear(input_size, input_size//2)),
                ("relu1", nn.ReLU()),
                ("Hidden Layer #2", nn.Linear(input_size//2, self.num_of_classes*3)),
                ("relu1", nn.ReLU()),
                ("Output Layer", nn.Linear(self.num_of_classes*3, self.num_of_classes)),
                 # ("relu", nn.ReLU()),
                ]
            )
        )
        

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        self.trainingLoss = []
        self.validaionLoss = []
        self.trainAcc = []
        self.validateAcc = []

    def initializeWeights(self):
        # print(self.parameters())
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def predict(self, x):
        # x = torch.flatten(x)
        return self.model(x)
    
    def getTrainingLoss(self):
        return self.trainingLoss
    
    def getValidationLoss(self):
        return self.validaionLoss

    def getTrainAccuracy(self):
        return self.trainAcc
        
    def getValAccuracy(self):
        return self.validateAcc


    def emptyLists(self):
        
        self.trainingLoss = []
        self.validaionLoss = []
        self.trainAcc = []
        self.validateAcc = []

    def train2(self, X, Y, batch_size = 64, epochs = 1):
        X, Y = shuffle(X, Y)
        
        for epoch in range(epochs):
            loss = 0.0
            total_loss = 0.0

            self.optimizer.zero_grad()
            for sample_idx in range(batch_size):
                i = epoch*batch_size + sample_idx
                y_predicted = self.predict(X[i])
                # print(f"\n\n {y_predicted} \n\n")

                target = torch.zeros(self.num_of_classes, dtype=torch.float32)
                target[Y[i]] = 1.0

                loss = self.loss_function(y_predicted.unsqueeze(0), torch.tensor([Y[i]]))
                # loss = self.loss_function(y_predicted, target)
                # print(f"Loss: { loss}")
                # print(f"\n\n {target} \n\n")

                loss.backward()
                total_loss += loss
            self.optimizer.step()

            # i = epoch*batch_size + sample_idx
            # y_predicted = self.predict(X[epoch*batch_size: (epoch+1)*batch_size])
            # loss = self.loss_function(Y[epoch*batch_size: (epoch+1)*batch_size], y_predicted)
            print(f"Total Loss: {total_loss}")
            # loss.backward()
            # self.optimizer.step()

    def train(self, X, Y, X_val = None, Y_val = None, batch_size=64, epochs=1):
        X, Y = shuffle(X, Y)

        for epoch in range(epochs):

            total_loss = 0.0
            total_val_loss = 0.0
            total_train_correct = 0
            total_val_correct = 0

            for i in range(0, len(X), batch_size):
                x = X[i:i+batch_size]
                target = Y[i:i+batch_size]
                y_pred = self.predict(x)
                    
                preds = torch.argmax(y_pred, dim=1)
                total_train_correct += (preds == target).sum().item()
                
                loss = self.loss_function(y_pred, target)
                total_loss += loss.item()
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                            
            if (X_val != None) and (Y_val != None):
                y_val_pred = self.predict(X_val)
                val_loss = self.loss_function(y_val_pred, Y_val)
                total_val_loss = val_loss.item()
                val_preds = torch.argmax(y_val_pred, dim=1)
                total_val_correct = (val_preds == Y_val).sum().item()

            trainAccuracy =  total_train_correct / (len(X)) 
            self.trainAcc.append(trainAccuracy)
            
            avrg_loss = total_loss/(len(X)//batch_size)
            
            self.trainingLoss.append(avrg_loss)
            
            if (X_val != None) and (Y_val != None):
                validationAccuracy =  total_val_correct / len(X_val) 
                self.validateAcc.append(validationAccuracy)
                self.validaionLoss.append(total_val_loss)
                print(f"-> Epoch {epoch+1}, Total Loss: {avrg_loss},  Total Evaluation Loss: {total_val_loss}")
                print(f"-> Model Training accuracy: {trainAccuracy},  Model Evaluation Accuracy: {validationAccuracy} \n")
            else:
                print(f"Epoch {epoch+1}, Total Loss: {total_loss}")
                print(f"-> Model Training accuracy: {trainAccuracy}\n")

            
