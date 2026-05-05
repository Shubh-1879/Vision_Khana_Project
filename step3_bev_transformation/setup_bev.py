"""
Step 3: BEV Transformation Setup
Convert natural scene thali images to bird's-eye-view (top-down) perspective

Key components:
1. Camera calibration from thali plate reference points
2. Perspective transformation using homography matrix
3. Warp thali images to 2D BEV
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path


def setup_bev_directories():
    """Create directory structure for BEV transformation"""
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    bev_dir = os.path.join(root_dir, 'step3_bev_transformation', 'data')
    
    dirs_to_create = [
        'natural_scenes/raw',
        'natural_scenes/annotated',
        'bev_transformed/images',
        'bev_transformed/masks',
        'calibration',
        'results'
    ]
    
    for dir_path in dirs_to_create:
        full_path = os.path.join(bev_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"[OK] Created: {full_path}")
    
    return bev_dir


def create_bev_readme(bev_dir):
    """Create README for BEV transformation"""
    
    readme_path = os.path.join(bev_dir, 'README.md')
    
    readme_content = """# Bird's Eye View (BEV) Transformation

## Overview
Transform natural scene thali images (with varying camera angles) to a standardized
bird's-eye-view (top-down) perspective.

## Why BEV?
- Normalizes images from different angles
- Makes object detection more consistent
- Enables direct food item area measurement
- Prepares data for nutritional analysis

## Directory Structure
```
data/
├── natural_scenes/
│   ├── raw/           # Raw images from different angles
│   └── annotated/     # Annotated with reference points
├── bev_transformed/
│   ├── images/        # BEV-warped images
│   └── masks/         # Thali plate segmentation masks
├── calibration/       # Camera calibration data
└── results/           # BEV transformation results
```

## Workflow

### Step 1: Collect Natural Scene Images
- Take photos of thali plates from different angles
- Include full thali plate in frame
- Ensure good lighting

### Step 2: Mark Reference Points
Use annotation tool to mark 4 corners of the thali plate:
- Top-left
- Top-right
- Bottom-left
- Bottom-right

### Step 3: Compute Homography Matrix
The script will:
1. Read reference points from annotations
2. Estimate homography matrix (camera perspective transformation)
3. Apply perspective warp to transform image to BEV

### Step 4: Detect Food Items in BEV
- Use Step 2 Faster R-CNN model on BEV images
- Get food items in normalized coordinate system

## Technical Details

### Homography Matrix
Maps 4 points in source image to 4 points in destination (BEV):
```
Source (natural angle):
  TL --- TR
  |      |
  BL --- BR

Destination (bird's eye view):
  (0,0) --- (W,0)
  |         |
  (0,H) --- (W,H)
```

### Perspective Transformation
Uses OpenCV's `getPerspectiveTransform()` and `warpPerspective()`

## Usage

```python
from bev_transformation import BEVTransformer

# Create transformer
bev = BEVTransformer()

# Transform single image
bev_image = bev.transform('natural_scenes/annotated/image1.json')

# Transform batch
bev.transform_batch('natural_scenes/annotated/')
```

## Expected Output
- BEV-warped images (normalized perspective)
- Transformation metadata (homography matrix, reference points)
- Detection results on BEV images
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"[OK] Created README: {readme_path}")


def main():
    print("Setting up BEV Transformation for Step 3...")
    print("="*60)
    
    bev_dir = setup_bev_directories()
    create_bev_readme(bev_dir)
    
    print("\n" + "="*60)
    print("[OK] BEV Transformation setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Collect natural scene thali images")
    print("2. Annotate reference points (4 corners of thali)")
    print("3. Run BEV transformation")
    print("4. Detect food items in BEV space")


if __name__ == '__main__':
    main()
