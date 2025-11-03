import torch
import torch.nn as nn
from collections import OrderedDict


class customNet(nn.Module):
    
    def __init__(self, input_size, num_of_classes, hidden_layers, learning_rate):
        super(customNet, self).__init__()

        self.num_of_classes = num_of_classes
        
        layers = []
        prev_size = input_size
        
        for idx, hidden_size in enumerate(hidden_layers):
            layers.append((f"Hidden Layer #{idx+1}", nn.Linear(prev_size, hidden_size)))
            layers.append((f"relu{idx+1}", nn.ReLU()))
            prev_size = hidden_size
        
        # Add output layer
        layers.append(("Output Layer", nn.Linear(prev_size, self.num_of_classes)))
        
        self.model = nn.Sequential(OrderedDict(layers))
        

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        self.trainingLoss = []
        self.validaionLoss = []
        self.trainingAccuracy = []
        self.validationAccuracy = []

    def initializeWeights(self):
        # print(self.parameters())
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        # x = torch.flatten(x)
        return self.model(x)
    
    def getTrainingLoss(self):
        return self.trainingLoss
    
    def getValidationLoss(self):
        return self.validaionLoss

    def getTrainingAccuracy(self):
        return self.trainingAccuracy
        
    def getValidationAccuracy(self):
        return self.validationAccuracy

    def emptyLists(self):
        
        self.trainingLoss = []
        self.validaionLoss = []
        self.trainingAccuracy = []
        self.validationAccuracy = []

    def fit(self, X_train, Y_train, X_val, Y_val, batch_size, epochs):
        
        super().train() 
        
        stats = {
            "train_loss": [], 
            "train_acc": [],
            "val_loss": [], 
            "val_acc": [],
            "grad_norms": []
        }

        for epoch in range(epochs):
            
            # Training
            num_batches = len(X_train) // batch_size
            
            epoch_grad_norms = []

            for i in range(num_batches):
                
                # Get batch
                start = i * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Forward pass
                Y_pred = self(X_batch)
                loss = self.loss_function(Y_pred, Y_batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Calculating Gradient Norm
                total_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norms.append(total_norm)

                # Update Weights
                self.optimizer.step()

            # Store grad norms from the last epoch for analysis
            if epoch == epochs - 1:
                stats["grad_norms"] = epoch_grad_norms

            # Evaluation
            train_loss, train_acc = self.evaluate(X_train, Y_train, batch_size)
            stats["train_loss"].append(train_loss)
            stats["train_acc"].append(train_acc)

            if X_val is not None and Y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, Y_val, batch_size)
                stats["val_loss"].append(val_loss)
                stats["val_acc"].append(val_acc)

            # Print metrics
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return stats
    
    def evaluate(self, X, Y, batch_size):
        self.eval()
        
        total_loss = 0
        total_correct = 0
        num_batches = len(X) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                Y_batch = Y[start:end]

                Y_pred = self(X_batch)
                loss = self.loss_function(Y_pred, Y_batch)

                total_loss += loss.item()

                preds = torch.argmax(Y_pred, dim=1)
                total_correct += (preds == Y_batch).sum().item()

        avg_loss = total_loss / num_batches
        accuracy = total_correct / len(X)

        return avg_loss, accuracy
