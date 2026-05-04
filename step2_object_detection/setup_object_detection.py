"""
Khana Dataset - Object Detection Setup (Step 2)
Setup for object detection on clean thali images
"""

import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json

def setup_object_detection_model(num_classes=81):  # 80 food classes + background
    """
    Setup Faster R-CNN model for object detection
    """
    # Load pretrained model weights
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def create_thali_dataset_structure():
    """
    Create directory structure for thali object detection dataset
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_dir = os.path.join(root_dir, 'step2_object_detection', 'thali_detection')

    dirs_to_create = [
        'images/train',
        'images/val',
        'images/test',
        'annotations/train',
        'annotations/val',
        'annotations/test',
        'models'
    ]

    for dir_path in dirs_to_create:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ Created: {full_path}")

    # Create README for dataset setup
    readme_content = """
    # Thali Object Detection Dataset Setup

    ## Required Files:
    1. Clean thali images (no background clutter)
    2. COCO-format annotations (.json files) with bounding boxes for each food item

    ## Directory Structure:
    thali_detection/
    ├── images/
    │   ├── train/     # Training images
    │   ├── val/       # Validation images
    │   └── test/      # Test images
    ├── annotations/
    │   ├── train/     # COCO annotations for training
    │   ├── val/       # COCO annotations for validation
    │   └── test/      # COCO annotations for test
    └── models/        # Trained detection models

    ## Annotation Format (COCO JSON):
    {
        "images": [...],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  // Food class ID (1-80)
                "bbox": [x, y, width, height],  // Bounding box
                "area": width * height,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "aloo_gobi"},
            // ... all 80 food classes
        ]
    }

    ## Next Steps:
    1. Add thali images to respective image folders
    2. Create COCO-format annotations using tools like:
       - LabelImg (https://github.com/tzutalin/labelImg)
       - VGG Image Annotator (VIA)
       - CVAT (https://cvat.org/)
    3. Run train_object_detection.py to train the model
    """

    readme_path = os.path.join(base_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✓ Created README: {readme_path}")

def main():
    print("Setting up Object Detection for Step 2...")
    print("="*50)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Create dataset structure
    create_thali_dataset_structure()

    # Setup model architecture
    print("\nSetting up Faster R-CNN model...")
    model = setup_object_detection_model()
    print("✓ Model architecture ready")

    # Save model template
    model_path = os.path.join(root_dir, 'step2_object_detection', 'thali_detection', 'models', 'faster_rcnn_template.pth')
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model template saved: {model_path}")

    print("\n" + "="*60)
    print("OBJECT DETECTION SETUP COMPLETE")
    print("="*60)
    print("Next steps:")
    print("1. Add clean thali images to step2_object_detection/thali_detection/images/")
    print("2. Create COCO-format annotations for bounding boxes")
    print("3. Run step2_object_detection/train_object_detection.py when data is ready")
    print("\nSee step2_object_detection/thali_detection/README.md for detailed instructions")

if __name__ == "__main__":
    main()