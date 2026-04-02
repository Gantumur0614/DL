import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import math
from skimage import io
from torch.utils.data import Dataset, DataLoader
from model import InceptionModel


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
        scaled_landmarks[:, 0] = scaled_landmarks[:, 0] / 178
        scaled_landmarks[:, 1] = scaled_landmarks[:, 1] / 218

        if self.transform:
            image_tr = self.transform(image)

        scaled_landmarks = scaled_landmarks.reshape(-1)
        sample = {'image': image, 'image_tr': image_tr,
                  'landmarks': landmarks, 'scaled_landmarks': scaled_landmarks}
        return sample


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

full_dataset = FaceLandmarksDataset(
    "/home/gantumur/Documents/DL/Lab456/data/list_landmarks_align_celeba.csv",
    "/home/gantumur/Documents/DL/Lab456/data/img_align_celeba/img_align_celeba",
    transform=transform
)

torch.manual_seed(42)
train_val_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_val_size
train_val_set, test_set = torch.utils.data.random_split(
    full_dataset, [train_val_size, test_size])

test_loader = DataLoader(test_set, batch_size=64,
                         shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

model = InceptionModel(aux=True, residual=True, num_classes=10).to(device)
weights_path = "/home/gantumur/Documents/DL/Lab456/models/best_inception_weights.pth"
model.load_state_dict(torch.load(weights_path))

model.eval()
criterion = nn.SmoothL1Loss()

print("Evaluating entire test dataset. This may take a minute...")
test_loss = 0.0
total_pixel_error = 0.0
total_landmarks = 0

with torch.no_grad():
    for i, batch_dict in enumerate(test_loader):
        images = batch_dict["image_tr"].to(device).float()
        targets = batch_dict["scaled_landmarks"].to(device).float()

        predictions = model(images)
        loss = criterion(predictions, targets)
        test_loss += loss.item()

        preds_cpu = predictions.cpu().numpy()
        targets_cpu = targets.cpu().numpy()

        batch_size = images.size(0)
        for b in range(batch_size):
            for p in range(5):

                pred_x = preds_cpu[b][2 * p] * 178
                pred_y = preds_cpu[b][2 * p + 1] * 218
                true_x = targets_cpu[b][2 * p] * 178
                true_y = targets_cpu[b][2 * p + 1] * 218

                dist = math.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                total_pixel_error += dist
                total_landmarks += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_loader)} batches...")

avg_test_loss = test_loss / len(test_loader)
avg_pixel_error = total_pixel_error / total_landmarks

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Average SmoothL1 Loss:  {avg_test_loss:.6f}")
print(f"Average Pixel Error:    {avg_pixel_error:.2f} pixels")
print("="*50)
