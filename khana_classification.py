"""
Khana Dataset Classification - Training Script
Trains a CNN to classify 80 Indian food classes with target accuracy >91%
"""

import sys
sys.stdout.flush()

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image, ImageFile
import warnings
from datetime import datetime

# Handle corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Palette images with Transparency")

def safe_pil_loader(path):
    """Safely load PIL image, handling corrupted files"""
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, IOError) as e:
        print(f"Warning: Could not load image {path}: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (224, 224), color=(128, 128, 128))

# Set data path
data_path = os.path.expanduser('~/Vision_Khana_Project/dataset/khana/')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("Loading dataset...")
dataset = ImageFolder(root=data_path, transform=transform, loader=safe_pil_loader)

# Split into train and val (80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f'Num classes: {len(dataset.classes)}')
print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

# Build a small CNN from scratch for an end-to-end pipeline
class SimpleKhanaCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout2d(0.2),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

num_classes = len(dataset.classes)
model = SimpleKhanaCNN(num_classes)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss, optimizer, scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f'Device: {device}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# Training functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

# Training loop - increased to 20 epochs for better convergence
print("\n" + "="*60)
print("Starting Training...")
print("="*60 + "\n")
sys.stdout.flush()

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    start_time = datetime.now()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    scheduler.step()
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60  # minutes
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% ({elapsed:.1f} min)')
    sys.stdout.flush()
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.expanduser('~/Vision_Khana_Project/best_model.pth'))
        print(f'  ✓ Saved best model with val_acc: {val_acc:.2f}%')
        sys.stdout.flush()

# Final validation accuracy
print("\n" + "="*60)
print("Training Complete!")
print("="*60 + "\n")
sys.stdout.flush()
_, final_val_acc = validate_epoch(model, val_loader, criterion, device)
print(f'Final Validation Accuracy: {final_val_acc:.2f}%')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
sys.stdout.flush()

if final_val_acc > 91:
    print('✓ Success: Above baseline (>91%)!')
else:
    print(f'✗ Below baseline (>91%), current: {final_val_acc:.2f}%')
sys.stdout.flush()
