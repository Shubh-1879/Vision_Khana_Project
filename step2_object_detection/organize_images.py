"""
Organize thali images: copy from project-thali_images-task2 to train/val folders
"""

import os
import shutil
from pathlib import Path

def copy_images_to_folders():
    """Copy raw thali images to train and val folders"""
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    thali_dir = os.path.join(root_dir, 'step2_object_detection', 'thali_detection')
    raw_images_dir = os.path.join(root_dir, 'project-thali_images-task2')
    
    train_img_dir = os.path.join(thali_dir, 'images', 'train')
    val_img_dir = os.path.join(thali_dir, 'images', 'val')
    
    if not os.path.exists(raw_images_dir):
        print(f"[INFO] Raw images directory not found: {raw_images_dir}")
        print(f"[INFO] Assuming images are already organized in {thali_dir}")
        
        # Verify that images exist
        train_count = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_count = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if train_count > 0 and val_count > 0:
            print(f"[OK] Found {train_count} training images and {val_count} validation images")
            return True
        else:
            print(f"[ERROR] No images found in {train_img_dir} or {val_img_dir}")
            return False
    
    # Get all image files from raw directory
    raw_images = sorted([f for f in os.listdir(raw_images_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not raw_images:
        print(f"[ERROR] No images found in {raw_images_dir}")
        return False
    
    print(f"Found {len(raw_images)} raw images")
    
    # Organize existing images
    train_xml_files = [f.replace('.xml', '.jpg') for f in os.listdir(os.path.join(thali_dir, 'annotations', 'train'))
                       if f.endswith('.xml')]
    val_xml_files = [f.replace('.xml', '.jpg') for f in os.listdir(os.path.join(thali_dir, 'annotations', 'val'))
                     if f.endswith('.xml')]
    
    print(f"\nTrain annotations: {len(train_xml_files)} images")
    print(f"Validation annotations: {len(val_xml_files)} images")
    
    # Copy training images
    print("\n" + "="*60)
    print("Copying TRAINING images...")
    print("="*60)
    for img_file in train_xml_files:
        src = os.path.join(raw_images_dir, img_file)
        dst = os.path.join(train_img_dir, img_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✓ {img_file}")
        else:
            print(f"⚠️  Not found: {src}")
    
    # Copy validation images
    print("\n" + "="*60)
    print("Copying VALIDATION images...")
    print("="*60)
    for img_file in val_xml_files:
        src = os.path.join(raw_images_dir, img_file)
        dst = os.path.join(val_img_dir, img_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[OK] {img_file}")
        else:
            print(f"[WARNING] Not found: {src}")
    
    print("\n" + "="*60)
    print("[OK] Images organized successfully!")
    print("="*60)
    print(f"Training images: {len(os.listdir(train_img_dir))}")
    print(f"Validation images: {len(os.listdir(val_img_dir))}")
    return True


if __name__ == '__main__':
    copy_images_to_folders()
