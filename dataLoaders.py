import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
def load_train_val():
    # Load data


    df = pd.read_csv("mnist_train.csv")

    # Separate features and labels
    x = df.iloc[:, 1:].values  # pixels
    y = df.iloc[:, 0].values  # labels

    # Normalize pixel values to [0, 1]
    x = x.astype(np.float32) / 255.0

    # Keep original shape (28x28) for later (for NN)
    X_images = x.reshape(-1, 1, 28, 28)

    # Stratified split (60/20/20)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.83, stratify=y, random_state=42)

    return x_train, x_val, y_train, y_val

def load_test():
    # Load data
    df = pd.read_csv("mnist_test.csv")

    # Separate features and labels
    x_test = df.iloc[:, 1:].values  # pixels
    y_test = df.iloc[:, 0].values  # labels

    # Normalize pixel values to [0, 1]
    x_test = x_test.astype(np.float32) / 255.0

    # Keep original shape (28x28) for later (for NN)
    X_images = x_test.reshape(-1, 1, 28, 28)

    return x_test, y_test