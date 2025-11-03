import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_train_val(path="mnist_train.csv"):
    # Load data
    df = pd.read_csv(path)

    # Separate features and labels
    x = df.iloc[:, 1:].values  # pixels
    y = df.iloc[:, 0].values  # labels

    # Keep original shape (28x28) for later (for NN)
    X_images = x.reshape(-1, 1, 28, 28)

    # Stratified split (60/20/20)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    return x_train, x_val, y_train, y_val

def load_test(path="mnist_test.csv"):
    # Load data
    df = pd.read_csv(path)

    # Separate features and labels
    x_test = df.iloc[:, 1:].values  # pixels
    y_test = df.iloc[:, 0].values  # labels

    # Keep original shape (28x28) for later (for NN)
    X_images = x_test.reshape(-1, 1, 28, 28)

    return x_test, y_test

def normalize(x):
    return x / 255.0

def cross_entropy_loss(y_batch,y_pred):
    return -torch.mean(y_batch * torch.log(y_pred + 1e-8) + (1 - y_batch) * torch.log(1 - y_pred + 1e-8))

def train_logistic(loader,lr,device,epochs = 100,plot=False):
# Initialize parameters
  train_losses = []
  W = torch.zeros(784, 1, dtype=torch.float32, requires_grad=True, device = device)
  b = torch.zeros(1, requires_grad=True, device = device)

  for epoch in range(epochs):
      epoch_loss = 0
      batch_loss = 0
      for x_batch, y_batch in loader:
          x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
          if W.grad is not None:
            W.grad.zero_()
            b.grad.zero_()
          # Forward pass
          y_pred = torch.sigmoid(x_batch @ W + b)
          # Backprop
          loss = cross_entropy_loss(y_batch=y_batch,y_pred=y_pred)
          batch_size = x_batch.size(0)  # actual number of samples in this batch
          epoch_loss += loss.item() * batch_size  # weight by batch size
          loss.backward()
          # Save copy for diff *after* backward but before update
          # Update
          with torch.no_grad():
              W.data -= lr * W.grad
              b.data -= lr * b.grad

      avg_loss = epoch_loss / len(loader.dataset)
      if (epoch/epochs*100)%25 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.10f}")

      # Average loss per epoch
      train_losses.append(avg_loss)
  if plot:
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
  print("Training done!")
  return W,b

def test_logistic_model(W, b, test_loader, device, cm_print = False):
    W = W.to(device)
    b = b.to(device)
    total = 0
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)

            # Forward pass
            y_pred = torch.sigmoid(x_batch @ W + b)

            # --- Compute Cross Entropy Loss ---
            loss = cross_entropy_loss(y_batch=y_batch,y_pred=y_pred)
            batch_size = x_batch.size(0)  # actual number of samples in this batch
            total_loss += loss.item() * batch_size  # weight by batch size

            # Convert probabilities to binary class labels
            predicted = (y_pred >= 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(y_batch.cpu())

    # Combine all predictions
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = correct / total
    mean_ce = total_loss / len(test_loader.dataset)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Cross-Entropy Loss: {mean_ce:.4f}")

    if cm_print:
      cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
      print("Confusion Matrix:\n", cm)
    return mean_ce

def validate(train_func,test_func,train_loader,val_loader,device,epochs = 100 , plot = False):
	lrs = [0.001,0.01,0.1,1.0]
	lrs = np.array(lrs)
	all_losses = list()
	best_lr = None
	for lr in lrs:
		print(f"Testing lr:{lr}")
		W, b = train_func(train_loader,lr=lr,device=device, epochs = epochs,plot=True)
		loss = test_func(W,b,val_loader, device) # Pass cm_print=False during validation
		all_losses.append(loss)
		print()
	all_losses = np.array(all_losses)
	best_idx = np.nanargmin(all_losses)
	best_lr= lrs[best_idx]
	best_loss = all_losses[best_idx]
	print(f"Best learning rate: {best_lr:.3e}    Validation CE: {best_loss:.6f}")
	if plot:
		plt.plot(lrs, all_losses)
		plt.xscale('log')
		plt.xlabel('learning rate')
		plt.ylabel('Validation CE')
		plt.title('Validation CE vs learning rate')
		plt.grid(True)
		plt.show()
	return best_lr

def softmax_manual(z):
    # subtract max for numerical stability
    z_exp = torch.exp(z - torch.max(z, dim=1, keepdim=True).values)
    return z_exp / torch.sum(z_exp, dim=1, keepdim=True)

def cross_entropy_softmax(y_pred, y_true):
    # y_true is [batch], y_pred is [batch, num_classes]
    # Gather predicted probability for the correct class
     return-torch.mean(torch.log(y_pred[torch.arange(len(y_true)), y_true] + 1e-8))

def train_softmax(train_loader,epochs,lr,device,plot=False):
  # Initialize parameters
  W = torch.zeros(784, 10, dtype=torch.float32, device=device, requires_grad=True)
  b = torch.zeros(10, dtype=torch.float32, device=device, requires_grad=True)
  train_losses, val_losses = [], []
  for epoch in range(epochs):
      epoch_loss = 0.0

      for x_batch, y_batch in train_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device, dtype=torch.long)

          # Forward pass
          y_pred = softmax_manual(x_batch @ W + b)

          # Loss
          loss = cross_entropy_softmax(y_pred, y_batch)

          # Backward
          loss.backward()

          # Update
          with torch.no_grad():
              W -= lr * W.grad
              b -= lr * b.grad
          W.grad.zero_()
          b.grad.zero_()

          epoch_loss += loss.item() * x_batch.size(0)

      train_loss = epoch_loss / len(train_loader.dataset)
      train_losses.append(train_loss)

      if (epoch/epochs*100)%25 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch+1}, Loss: {train_loss:.10f}")
        #print("Pred mean:", y_pred.mean().item())
        #print(f"Grad mean: {W.grad.abs().mean().item():.10f}")
  if plot:
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
  print("Training done!")
  return W, b

def test_softmax(W, b, test_loader, device, cm_print = True):
    W = W.to(device)
    b = b.to(device)
    total = 0
    correct = 0
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device, dtype=torch.long)

            y_pred = softmax_manual(x_batch @ W + b)

            loss = cross_entropy_softmax(y_pred, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    accuracy = correct / total
    mean_loss = total_loss / len(test_loader.dataset)

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Cross-Entropy Loss: {mean_loss:.4f}")

    if cm_print:
    # Confusion Matrix
      cm = confusion_matrix(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())
      print("Confusion Matrix:\n", cm)
    return mean_loss