import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from skimage import io, transform
import os
import torch.optim as optim
import itertools
from model import InceptionModel 
import logging 

log_dir = "/home/gantumur/Documents/DL/Lab456/logs"

log_path = os.path.join(log_dir, "batch_overfitting.log")


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
logging.info("Starting Batch overfitting Session")
logging.info("--------------------------------")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None):
        self.landmarks_frame = []
        file = open(csv_file, "r")
        while True:
            content = file.readline()
            if not content:
                break
            self.landmarks_frame.append(content.split(","))
        file.close()
        self.landmarks_frame = self.landmarks_frame[2:]

        self.img_root_dir = img_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(
            self.img_root_dir, self.landmarks_frame[idx][0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame[idx][1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0], scaled_landmarks[:,
                                                 1] = scaled_landmarks[:, 0] / 178, scaled_landmarks[:, 1] / 218
        if self.transform:
            image_tr = self.transform(image)
        scaled_landmarks = scaled_landmarks.reshape(-1)
        sample = {'image': image, 'image_tr': image_tr,
                  'landmarks': landmarks, 'scaled_landmarks': scaled_landmarks}

        return sample


#1. Data Loading 
full_dataset = FaceLandmarksDataset("/home/gantumur/Documents/DL/Lab456/data/list_landmarks_align_celeba.csv",
                                    "/home/gantumur/Documents/DL/Lab456/data/img_align_celeba/img_align_celeba", transform=transform)

train_val_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_val_size
train_val_set, test_set = torch.utils.data.random_split(
    full_dataset, [train_val_size, test_size])

train_size = int(0.8 * len(train_val_set))
val_size = len(train_val_set) - train_size
train_set, val_set = torch.utils.data.random_split(
    train_val_set, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=16,
    shuffle=True # should be False fatal mistake 
)

batch_dict = next(iter(train_loader))

X_single_batch = batch_dict["image_tr"]
y_single_batch = batch_dict["scaled_landmarks"]


learning_rates = [1e-3, 5e-4, 1e-4, 1e-5]
optimizers = ["Adam", "SGD_Momentum"]

loss_functions = {
    "MSE": nn.MSELoss(),
    "SmoothL1": nn.SmoothL1Loss()
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

X_single_batch = X_single_batch.to(device).float()
y_single_batch = y_single_batch.to(device).float()

epochs_per_test = 300
best_loss = float("inf")
best_combo = None

for lr, opt_name, loss_name in itertools.product(learning_rates, optimizers, loss_functions):
    model = InceptionModel(aux=True, residual=True, num_classes=10).to(device)
    model.train()

    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion = loss_functions[loss_name]
    aux_weight = 0.3
    final_loss = 0.0

    for epoch in range(epochs_per_test):
        optimizer.zero_grad()
        main_out, aux_out = model(X_single_batch)

        loss_main = criterion(main_out, y_single_batch)
        loss_aux = criterion(aux_out, y_single_batch)

        total_loss = loss_main + (aux_weight * loss_aux)

        total_loss.backward()
        optimizer.step()

        final_loss = total_loss.item()
    print(
        f"Tested: LR={lr}, Opt={opt_name}, Loss={loss_name} | Final Loss: {final_loss:.6f}")
    
    logging.info(
        f"Tested: LR={lr}, Opt={opt_name}, Loss={loss_name} | Final Loss: {final_loss:.6f}")
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_combo = {'LR': lr, 'Optimizer': opt_name, 'Loss': loss_name}

    del model
    del optimizer
    torch.cuda.empty_cache()

logging.info(f"Best Combination Found: {best_combo}")
logging.info(f"Lowest Loss Achieved: {best_loss:.6f}")

print(f"Best Combination Found: {best_combo}")
print(f"Lowest Loss Achieved: {best_loss:.6f}")
