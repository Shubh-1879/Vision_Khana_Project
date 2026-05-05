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
