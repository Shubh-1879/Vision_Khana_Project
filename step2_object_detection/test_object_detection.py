"""
Khana Dataset - Object Detection Sanity Check
Verify the detection training setup and model forward pass.
"""

import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from PIL import Image


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_detection_model(num_classes=81):
    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def sanity_check_data_dirs():
    root = get_project_root()
    base_dir = os.path.join(root, 'step2_object_detection', 'thali_detection')
    required_dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'images', 'test'),
        os.path.join(base_dir, 'annotations', 'train'),
        os.path.join(base_dir, 'annotations', 'val'),
        os.path.join(base_dir, 'annotations', 'test')
    ]
    missing = [d for d in required_dirs if not os.path.isdir(d)]
    if missing:
        print('[WARNING] Missing required directories:')
        for path in missing:
            print('  -', path)
        return False
    print('[OK] Object detection directories are present')
    return True


def sanity_check_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_detection_model(num_classes=81)
    model = model.to(device)
    model.eval()

    dummy_image = torch.rand(3, 512, 512).to(device)
    with torch.no_grad():
        output = model([dummy_image])

    if isinstance(output, list) and len(output) == 1:
        print('[OK] Model forward pass succeeded')
        print(f'  Output keys: {list(output[0].keys())}')
        return True
    print('[ERROR] Model forward pass failed')
    return False


def main():
    print('STEP 2 SANITY CHECK')
    print('='*50)
    data_ok = sanity_check_data_dirs()
    model_ok = sanity_check_model()

    if data_ok and model_ok:
        print('\n[OK] Step 2 object detection framework is ready')
    else:
        print('\n[WARNING] Fix the issues above before training')


if __name__ == '__main__':
    main()