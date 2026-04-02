import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import cupy as cp
import time
import os
import logging  
from model import *  
import pandas as pd 

log_dir = "/home/gantumur/Documents/DL/Lab123/log"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "bigger_model_training.log")  


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path),   
        logging.StreamHandler()          
    ]
)

logging.info("--------------------------------")
logging.info("Starting Training Session")
logging.info("--------------------------------")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

fullset = datasets.MNIST(root="./data", train=True,
                         download=True, transform=transform)
train_size = int(len(fullset) * 0.8)
val_size = len(fullset) - train_size
trainset, valset = random_split(fullset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
valloader = DataLoader(valset, batch_size=128, shuffle=False)

model = ConvNET([
    ConvLayer(in_channels=1, out_channels=32, filter_dim=3, pad=0, alpha=0.1),
    ReLU(),
    MaxPool(pool_size=2, stride=2),

    ConvLayer(in_channels=32, out_channels=64, filter_dim=3, pad=0, alpha=0.1),
    ReLU(),
    MaxPool(pool_size=2, stride=2),

    Flatten(),

    Linear_Layer(input_dim=64*5*5, output_dim=128, alpha=0.1),
    ReLU(),

    Linear_Layer(input_dim=128, output_dim=10, alpha=0.1)
])

loss_fn = SoftMaxCrossEntropy()
epochs = 30
min_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(epochs):
    epoch_start = time.time()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        x_batch = cp.asarray(data.numpy())
        y_batch = cp.asarray(target.numpy())

        logits = model.forward(x_batch)
        loss = loss_fn.forward(logits, y_batch)
        train_loss += loss.item()

        grad = loss_fn.backprop()
        model.backward(grad)
        model.update()

        predictions = cp.argmax(logits, axis=1)
        train_correct += (predictions == y_batch).sum().item()
        train_total += y_batch.shape[0]

    avg_train_loss = train_loss / len(trainloader)
    train_acc = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)

    val_loss = 0
    val_correct = 0
    val_total = 0

    for data, target in valloader:
        x_val = cp.asarray(data.numpy())
        y_val = cp.asarray(target.numpy())

        logits = model.forward(x_val)
        loss = loss_fn.forward(logits, y_val)
        val_loss += loss.item()

        predictions = cp.argmax(logits, axis=1)
        val_correct += (predictions == y_val).sum().item()
        val_total += y_val.shape[0]

    avg_val_loss = val_loss / len(valloader)
    val_losses.append(avg_val_loss)
    val_acc = 100 * val_correct / val_total

    logging.info(f"Epoch {epoch+1}/{epochs}")
    logging.info(
        f"   Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
    logging.info(f"   Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

    if avg_val_loss < min_val_loss:
        logging.info(f"   --> Saving Best Model ({avg_val_loss:.4f})")
        min_val_loss = avg_val_loss

        checkpoint = {}
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'filters'):
                checkpoint[f'layer_{i}_filters'] = cp.asnumpy(layer.filters)
                checkpoint[f'layer_{i}_bias'] = cp.asnumpy(layer.bias)
            elif hasattr(layer, 'theta'):
                checkpoint[f'layer_{i}_theta'] = cp.asnumpy(layer.theta)
                checkpoint[f'layer_{i}_bias'] = cp.asnumpy(layer.bias)

        torch.save(
            checkpoint, "/home/gantumur/Documents/DL/Lab123/models/bigger_best_model_weights.pth")

logging.info("Training Complete.")

df = pd.DataFrame({
    "train_losses": train_losses,
    "val_losses": val_losses
})

df.to_csv("/home/gantumur/Documents/DL/Lab123/loss/loss_info.csv", index=False)
