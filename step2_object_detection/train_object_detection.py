"""
Khana Dataset - Object Detection Training (Step 2)
Train Faster R-CNN model on thali images with food item bounding boxes
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


def load_class_names():
    """Load all food class names from dataset/labels.txt"""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    labels_path = os.path.join(root_dir, 'dataset', 'labels.txt')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'labels.txt not found at {labels_path}')
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

class CocoDataset(Dataset):
    """COCO format dataset for object detection"""
    
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.coco_data['images'])
    
    def __getitem__(self, idx):
        img_info = self.coco_data['images'][idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            # Handle images with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.functional.to_tensor(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target


def get_detection_model(num_classes=None):
    """Create Faster R-CNN model for detection"""
    if num_classes is None:
        class_names = load_class_names()
        num_classes = len(class_names) + 1

    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    
    # Replace classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix({'loss': losses.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, device, epoch, num_epochs):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Get predictions
        predictions = model(images)
        
        pbar.set_postfix({'images': len(images)})
    
    return total_loss


def train_detection_model(data_dir, num_epochs=10, batch_size=4, lr=0.005):
    """Main training function"""
    
    print("="*60)
    print("OBJECT DETECTION TRAINING (Step 2)")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    train_img_dir = os.path.join(data_dir, 'images', 'train')
    train_ann_file = os.path.join(data_dir, 'annotations', 'train', 'instances_train.json')
    val_img_dir = os.path.join(data_dir, 'images', 'val')
    val_ann_file = os.path.join(data_dir, 'annotations', 'val', 'instances_val.json')
    
    # Check if data exists
    if not os.path.exists(train_ann_file):
        print(f"\n❌ Training annotations not found at: {train_ann_file}")
        print("Please create COCO-format annotations first using annotation_helper.py")
        return None
    
    print(f"\n📁 Data paths:")
    print(f"   Train images: {train_img_dir}")
    print(f"   Train annotations: {train_ann_file}")
    print(f"   Val images: {val_img_dir}")
    print(f"   Val annotations: {val_ann_file}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    train_dataset = CocoDataset(train_img_dir, train_ann_file, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn,
                             num_workers=0)
    
    val_dataset = CocoDataset(val_img_dir, val_ann_file, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn,
                           num_workers=0)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create model
    class_names = load_class_names()
    num_classes = len(class_names) + 1
    print(f"\n🧠 Creating model with {num_classes} classes ({len(class_names)} food labels + background)")
    model = get_detection_model(num_classes=num_classes)
    model = model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    print("\n🚀 Starting training...\n")
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    model_dir = os.path.join(data_dir, 'models')
    best_model_path = os.path.join(model_dir, 'best_detection_model.pth')
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, num_epochs)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, device, epoch, num_epochs)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved: {best_model_path}")
    
    # Save final report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Faster R-CNN ResNet50 FPN",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": str(device),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_loss,
        "best_model_path": best_model_path
    }
    
    report_path = os.path.join(model_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved: {best_model_path}")
    print(f"Report saved: {report_path}")
    
    return model


if __name__ == "__main__":
    import sys
    
    # Use provided data directory or default
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root_dir, 'step2_object_detection', 'thali_detection')
    
    model = train_detection_model(data_dir, num_epochs=10, batch_size=4, lr=0.005)
