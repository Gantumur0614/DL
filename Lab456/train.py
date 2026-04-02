import torch
import torch.nn as nn 
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import torch.optim as optim
from model import InceptionModel
import logging
import time 
import pandas as pd 


log_dir = "/home/gantumur/Documents/DL/Lab456/logs"

log_path = os.path.join(log_dir, "training.log")


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


# 1. Data Loading
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


train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


epochs = 30
LR = 5e-4
aux_weight = 0.3
min_val_loss = float("inf")
train_losses = [] 
val_losses = []  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
logging.info(f"Device: {device}")

model = InceptionModel(
    aux=True,
    residual=True,
    num_classes=10
).to(device)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.SmoothL1Loss()

for epoch in range(epochs):
    epoch_start = time.time() 

    model.train() 
    train_loss = 0.0 

    for batch_dict in train_loader:
        x_batch = batch_dict["image_tr"].to(device).float()
        y_batch = batch_dict["scaled_landmarks"].to(device).float()

        optimizer.zero_grad()

        main_out, aux_out = model(x_batch)

        loss_main = criterion(main_out, y_batch)
        loss_aux = criterion(aux_out, y_batch)
        loss = loss_main + (aux_weight * loss_aux)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validation 
    model.eval()
    val_loss = 0.0 

    with torch.no_grad():
        for batch_dict in val_loader:
            x_val = batch_dict["image_tr"].to(device).float()
            y_val = batch_dict["scaled_landmarks"].to(device).float()

            main_out = model(x_val)
            
            loss = criterion(main_out, y_val)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    epoch_time = time.time() - epoch_start 

    logging.info(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s]")
    logging.info(f"   Train Loss: {avg_train_loss:.4f}")
    logging.info(f"   Val   Loss: {avg_val_loss:.4f}")

    if avg_val_loss < min_val_loss:
        logging.info(f"   --> Saving Best Model ({avg_val_loss:.4f})")
        min_val_loss = avg_val_loss

        torch.save(model.state_dict(
        ), "/home/gantumur/Documents/DL/Lab456/models/best_inception_weights.pth")

logging.info("Training Complete.")

df = pd.DataFrame({
    "train_losses": train_losses,
    "val_losses": val_losses
})
df.to_csv("/home/gantumur/Documents/DL/Lab456/loss/loss_info.csv", index=False)








