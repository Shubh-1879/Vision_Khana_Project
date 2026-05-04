
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
    