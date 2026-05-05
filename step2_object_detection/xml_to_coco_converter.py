"""
Convert LabelImg XML annotations to COCO JSON format
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def load_class_names():
    """Load the 80 food class names from labels.txt"""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    labels_path = os.path.join(root_dir, 'dataset', 'labels.txt')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'labels.txt not found at {labels_path}')
    with open(labels_path, 'r', encoding='utf-8') as f:
        classes = [line.strip().lower().replace(' ', '_') for line in f if line.strip()]
    return {cls: idx + 1 for idx, cls in enumerate(classes)}


def xml_to_coco(xml_dir, image_dir, output_json_path, class_mapping):
    """Convert XML annotations in a directory to COCO format"""
    
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Build categories list
    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        coco_data['categories'].append({
            'id': class_id,
            'name': class_name,
            'supercategory': 'food'
        })
    
    # Find all XML files
    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
    
    if not xml_files:
        print(f"[WARNING] No XML files found in {xml_dir}")
        return coco_data
    
    annotation_id = 1
    image_id = 1
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)
            
            # Check if image file exists
            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found: {img_path}, skipping...")
                continue
            
            # Add image info
            coco_data['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': width,
                'height': height
            })
            
            # Process objects
            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower().replace(' ', '_')
                
                # Handle class name variations
                if class_name not in class_mapping:
                    print(f"[WARNING] Unknown class '{class_name}' in {xml_file}, skipping object...")
                    continue
                
                class_id = class_mapping[class_name]
                
                # Get bounding box
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Convert to COCO format [x, y, width, height]
                box_width = xmax - xmin
                box_height = ymax - ymin
                
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': class_id,
                    'bbox': [xmin, ymin, box_width, box_height],
                    'area': box_width * box_height,
                    'iscrowd': 0
                })
                
                annotation_id += 1
            
            image_id += 1
            print(f"[OK] Processed: {xml_file}")
            
        except Exception as e:
            print(f"[ERROR] Error processing {xml_file}: {e}")
            continue
    
    # Save COCO JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n[OK] Saved COCO JSON: {output_json_path}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    return coco_data


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    thali_dir = os.path.join(root_dir, 'step2_object_detection', 'thali_detection')
    
    # Load class mapping
    print("Loading food class names...")
    class_mapping = load_class_names()
    print(f"[OK] Loaded {len(class_mapping)} food classes")
    
    # Convert training annotations
    print("\n" + "="*60)
    print("Converting TRAINING annotations...")
    print("="*60)
    train_xml_dir = os.path.join(thali_dir, 'annotations', 'train')
    train_img_dir = os.path.join(thali_dir, 'images', 'train')
    train_json_path = os.path.join(thali_dir, 'annotations', 'train', 'instances_train.json')
    
    xml_to_coco(train_xml_dir, train_img_dir, train_json_path, class_mapping)
    
    # Convert validation annotations
    print("\n" + "="*60)
    print("Converting VALIDATION annotations...")
    print("="*60)
    val_xml_dir = os.path.join(thali_dir, 'annotations', 'val')
    val_img_dir = os.path.join(thali_dir, 'images', 'val')
    val_json_path = os.path.join(thali_dir, 'annotations', 'val', 'instances_val.json')
    
    xml_to_coco(val_xml_dir, val_img_dir, val_json_path, class_mapping)
    
    print("\n" + "="*60)
    print("[OK] Conversion complete!")
    print("="*60)


if __name__ == '__main__':
    main()
