"""
Khana Dataset Classification - ResNet50 Fine-tuning Script
Fine-tunes pretrained ResNet50 to classify 80 Indian food classes.
Updated for HPC: AdamW, 30 Epochs, num_workers=32, pin_memory=True, and fixed val leakage.
"""

import sys
sys.stdout.flush()

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset
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

# Resolve repository root relative to this script
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set data path
data_path = os.path.join(root_dir, 'dataset', 'khana')

# Define SEPARATE transforms to fix data leakage
# Train gets heavy augmentations, Val gets only resize/normalize
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Wrapper to apply transforms AFTER splitting
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Load base dataset (NO transforms yet)
print("Loading dataset metadata...")
base_dataset = ImageFolder(root=data_path, loader=safe_pil_loader)

# Calculate split sizes
train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

# Generate random indices for the split
generator = torch.Generator().manual_seed(42) # Seed for reproducibility
indices = torch.randperm(len(base_dataset), generator=generator).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create subsets and wrap them with their respective transforms
train_subset = Subset(base_dataset, train_indices)
val_subset = Subset(base_dataset, val_indices)

train_dataset = DatasetWrapper(train_subset, transform=train_transform)
val_dataset = DatasetWrapper(val_subset, transform=val_transform)

# Data loaders - HPC OPTIMIZED
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=32, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=32, 
    pin_memory=True
)

print(f'Num classes: {len(base_dataset.classes)}')
print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

# Load pretrained ResNet50 and fine-tune for 80 classes
print("Loading pretrained ResNet50...")
num_classes = len(base_dataset.classes)
model = torchvision.models.resnet50(pretrained=False)  # Load architecture only

# Load pretrained weights from local file
weights_path = os.path.join(root_dir, 'pretrained_weights', 'resnet50-19c8e357.pth')
if os.path.exists(weights_path):
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
else:
    print(f"WARNING: Pretrained weights not found at {weights_path}")
    print("Please download weights using: python download_weights.py")

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
# Only optimize unfrozen parameters
params_to_optimize = [p for p in model.parameters() if p.requires_grad]

# UPDATED: AdamW optimizer for better weight decay handling
optimizer = torch.optim.AdamW(params_to_optimize, lr=0.0001, weight_decay=0.01)

# UPDATED: Adjusted step_size to 10 to accommodate 30 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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

# Training loop
print("\n" + "="*60)
print("Starting ResNet50 Fine-tuning Training...")
print("="*60 + "\n")
sys.stdout.flush()

# UPDATED: 30 Epochs
num_epochs = 30
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
        torch.save(model.state_dict(), os.path.join(root_dir, 'step1_classification', 'best_model_resnet.pth'))
        print(f'  ✓ Saved best model with val_acc: {val_acc:.2f}%')
        sys.stdout.flush()

# Final validation accuracy
print("\n" + "="*60)
print("ResNet50 Fine-tuning Complete!")
print("="*60 + "\n")
sys.stdout.flush()

# Load the best weights before final eval just to be sure we report the peak
model.load_state_dict(torch.load(os.path.join(root_dir, 'step1_classification', 'best_model_resnet.pth')))
_, final_val_acc = validate_epoch(model, val_loader, criterion, device)

print(f'Final Best Validation Accuracy: {final_val_acc:.2f}%')
sys.stdout.flush()

if final_val_acc > 91:
    print('✓ Success: Above baseline (>91%)!')
else:
    print(f'✗ Below baseline (>91%), current: {final_val_acc:.2f}%')
sys.stdout.flush()