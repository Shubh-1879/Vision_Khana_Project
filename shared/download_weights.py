"""
Download pretrained ResNet50 weights for offline use on HPC
"""
import torch
import torchvision
import os

# Resolve repository root relative to this script
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create directory for weights
weights_dir = os.path.join(root_dir, 'pretrained_weights')
os.makedirs(weights_dir, exist_ok=True)

print("Downloading ResNet50 pretrained weights...")
model = torchvision.models.resnet50(pretrained=True)

# Save to local directory
weights_path = os.path.join(weights_dir, 'resnet50-19c8e357.pth')
torch.save(model.state_dict(), weights_path)

print(f"✓ Weights downloaded and saved to: {weights_path}")
print("Transfer this file to HPC or use locally for training")
