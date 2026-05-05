# Bird's Eye View (BEV) Transformation

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
