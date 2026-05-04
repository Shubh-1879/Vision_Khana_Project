"""
Khana Dataset - Object Detection Annotation Helper
Generate COCO-format annotation templates and validate COCO JSON files.
"""

import json
import os
from pathlib import Path


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_class_names():
    """Load the 80 food class names from labels.txt"""
    root = get_project_root()
    labels_path = os.path.join(root, 'Khana Dataset', 'labels.txt')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f'labels.txt not found at {labels_path}')
    with open(labels_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def build_coco_categories(class_names):
    """Build COCO categories list from food class names"""
    categories = []
    for idx, class_name in enumerate(class_names, start=1):
        categories.append({
            'id': idx,
            'name': class_name.replace(' ', '_'),
            'supercategory': 'food'
        })
    return categories


def generate_coco_template(image_dir, output_path):
    """Generate a COCO annotation template skeleton for the given image folder"""
    image_dir = os.path.expanduser(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image directory not found: {image_dir}')

    class_names = load_class_names()
    categories = build_coco_categories(class_names)

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    images = []
    for idx, file_name in enumerate(image_files, start=1):
        images.append({
            'id': idx,
            'file_name': file_name,
            'width': 1024,
            'height': 1024
        })

    coco_json = {
        'images': images,
        'annotations': [],
        'categories': categories
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_json, f, indent=2)

    print(f'✓ Generated COCO template: {output_path}')
    print(f'  Images: {len(images)}')
    print(f'  Categories: {len(categories)}')
    return output_path


def validate_coco_annotations(annotation_path, image_dir):
    """Validate a COCO annotation JSON file against image files"""
    annotation_path = os.path.expanduser(annotation_path)
    image_dir = os.path.expanduser(image_dir)

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f'Annotation file not found: {annotation_path}')
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Image directory not found: {image_dir}')

    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    required_keys = {'images', 'annotations', 'categories'}
    if not required_keys.issubset(data.keys()):
        raise ValueError('COCO JSON must contain images, annotations, and categories keys')

    image_files = {img['file_name'] for img in data['images']}
    missing = [fn for fn in image_files if not os.path.exists(os.path.join(image_dir, fn))]
    if missing:
        raise ValueError(f'Missing image files referenced in annotations: {missing[:10]}')

    print('✓ COCO annotation file validated successfully')
    print(f'  Images: {len(data["images"])}')
    print(f'  Annotations: {len(data["annotations"])}')
    print(f'  Categories: {len(data["categories"])}')
    return True


def main():
    root = get_project_root()
    thali_dir = os.path.join(root, 'step2_object_detection', 'thali_detection')
    train_image_dir = os.path.join(thali_dir, 'images', 'train')
    output_template = os.path.join(thali_dir, 'annotations', 'train', 'instances_train.json')

    generate_coco_template(train_image_dir, output_template)

    print('\nNext: annotate the images and save the bounding boxes in the same JSON file.')
    print('After annotation, use validate_coco_annotations() to check the file.')


if __name__ == '__main__':
    main()