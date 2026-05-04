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

# Load pretrained ResNet50 and fine-tune for 80 classes
print("Loading pretrained ResNet50...")
num_classes = len(dataset.classes)
model = torchvision.models.resnet50(pretrained=True)

# Freeze early layers for fine-tuning (layers 1 and 2)
print("Freezing early layers for fine-tuning...")
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Replace final classification layer for 80 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss, optimizer, scheduler
criterion = torch.nn.CrossEntropyLoss()
# Only optimize unfrozen parameters (layer3, layer4, and fc)
params_to_optimize = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params_to_optimize, lr=0.0001)  # Lower LR for fine-tuning
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

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

num_epochs = 3
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
