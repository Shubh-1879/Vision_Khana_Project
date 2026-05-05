"""
BEV Transformation Module
Transform natural scene thali images to bird's-eye-view
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class BEVTransformer:
    """Transform thali images from natural angle to bird's-eye-view (BEV)"""
    
    def __init__(self, output_size=(512, 512)):
        """
        Initialize BEV transformer
        
        Args:
            output_size: (width, height) of output BEV image
        """
        self.output_size = output_size
        self.homography_matrix = None
    
    
    def estimate_homography(self, src_points):
        """
        Estimate homography matrix from reference points
        
        Args:
            src_points: list of 4 points [TL, TR, BL, BR] in source image
        
        Returns:
            homography matrix
        """
        # Define destination points (BEV - top-down view)
        dst_points = np.array([
            [0, 0],                           # Top-left
            [self.output_size[0], 0],         # Top-right
            [0, self.output_size[1]],         # Bottom-left
            [self.output_size[0], self.output_size[1]]  # Bottom-right
        ], dtype=np.float32)
        
        src_points = np.array(src_points, dtype=np.float32)
        
        # Compute homography matrix
        H, status = cv2.findHomography(src_points, dst_points)
        
        return H
    
    
    def transform_image(self, image, homography_matrix):
        """
        Apply perspective transformation to image
        
        Args:
            image: input image (BGR)
            homography_matrix: 3x3 transformation matrix
        
        Returns:
            BEV-transformed image
        """
        bev_image = cv2.warpPerspective(
            image,
            homography_matrix,
            self.output_size
        )
        
        return bev_image
    
    
    def transform_from_annotation(self, image_path, annotation_json):
        """
        Transform image using annotation file with reference points
        
        Args:
            image_path: path to image file
            annotation_json: path to JSON file with reference points
                            Format: {"image_path": "", "reference_points": [[x1,y1], [x2,y2], ...]}
        
        Returns:
            BEV-transformed image
        """
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Load reference points
        if not os.path.exists(annotation_json):
            raise FileNotFoundError(f"Annotation file not found: {annotation_json}")
        
        with open(annotation_json, 'r') as f:
            data = json.load(f)
        
        ref_points = data.get('reference_points', [])
        if len(ref_points) != 4:
            raise ValueError(f"Expected 4 reference points, got {len(ref_points)}")
        
        # Estimate homography
        H = self.estimate_homography(ref_points)
        
        # Transform image
        bev_image = self.transform_image(image, H)
        
        return bev_image, H
    
    
    def detect_food_in_bev(self, bev_image, model, class_names, device='cpu'):
        """
        Detect food items in BEV image using Faster R-CNN
        
        Args:
            bev_image: BEV-transformed image (BGR)
            model: Faster R-CNN model
            class_names: list of food class names
            device: torch device
        
        Returns:
            list of detections: [class_name, bbox, confidence]
        """
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
        image_tensor = torchvision.transforms.functional.to_tensor(image_rgb)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Run detection
        model.eval()
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Parse results
        detections = []
        pred = predictions[0]
        
        for i in range(len(pred['boxes'])):
            box = pred['boxes'][i].cpu().numpy()
            label = pred['labels'][i].item()
            score = pred['scores'][i].item()
            
            if score < 0.5:  # Confidence threshold
                continue
            
            class_name = class_names[label - 1] if 0 <= label - 1 < len(class_names) else f"class_{label}"
            
            detections.append({
                'class': class_name,
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                'confidence': float(score)
            })
        
        return detections

    def _load_annotation(self, annotation_json):
        annotation_json = os.path.expanduser(annotation_json)
        if not os.path.exists(annotation_json):
            raise FileNotFoundError(f"Annotation file not found: {annotation_json}")

        with open(annotation_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ref_points = data.get('reference_points')
        if not isinstance(ref_points, list) or len(ref_points) != 4:
            raise ValueError(f"Annotation JSON must contain 4 reference_points, got {ref_points}")

        image_path = data.get('image_path') or data.get('image') or data.get('file_name')
        if image_path is None:
            base_path = os.path.splitext(annotation_json)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = base_path + ext
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        if image_path is None:
            raise ValueError(
                'Annotation JSON must include image_path or be named as <image_name>.json next to the image file'
            )

        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(annotation_json), image_path)

        return annotation_json, image_path, ref_points, data

    def transform(self, annotation_json, output_path=None, save_homography=False, homography_output_path=None):
        annotation_json, image_path, ref_points, _ = self._load_annotation(annotation_json)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        H = self.estimate_homography(ref_points)
        bev_image = self.transform_image(image, H)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, bev_image)

        if save_homography and homography_output_path:
            os.makedirs(os.path.dirname(homography_output_path), exist_ok=True)
            with open(homography_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_path': image_path,
                    'reference_points': ref_points,
                    'homography': H.tolist()
                }, f, indent=2)

        return bev_image, H

    def transform_batch(self, annotation_dir, output_dir=None, save_homography=False):
        annotation_dir = os.path.expanduser(annotation_dir)
        if not os.path.isdir(annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(annotation_dir, '..', '..', 'bev_transformed', 'images'))

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for file_name in sorted(os.listdir(annotation_dir)):
            if not file_name.lower().endswith('.json'):
                continue

            annotation_path = os.path.join(annotation_dir, file_name)
            output_image_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.jpg')
            homography_path = None
            if save_homography:
                homography_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '_homography.json')

            bev_image, H = self.transform(
                annotation_path,
                output_path=output_image_path,
                save_homography=save_homography,
                homography_output_path=homography_path
            )

            results.append({
                'annotation': annotation_path,
                'output_image': output_image_path,
                'homography': H.tolist()
            })

        return results

    def compute_food_area(self, bbox, pixel_to_cm_ratio=1.0):
        """
        Compute food item area from bounding box
        
        Args:
            bbox: [x1, y1, x2, y2]
            pixel_to_cm_ratio: conversion factor (pixels per cm)
        
        Returns:
            area in cm²
        """
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1
        area_px = width_px * height_px
        
        # Convert to cm²
        area_cm2 = area_px / (pixel_to_cm_ratio ** 2)
        
        return area_cm2


def main():
    print("BEV Transformation Module")
    print("="*60)
    print("This module provides perspective transformation for thali images")
    print("\nUsage:")
    print("  from bev_transformation import BEVTransformer")
    print("  bev = BEVTransformer(output_size=(512, 512))")
    print("  bev_image, H = bev.transform_from_annotation('image.jpg', 'annotation.json')")


if __name__ == '__main__':
    main()
