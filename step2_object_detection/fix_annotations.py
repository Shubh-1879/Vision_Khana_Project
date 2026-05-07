import json
import os

# === CHANGE THIS LIST IF NEEDED ===
TARGET_CLASSES = [
    "aloo paratha", "chapati", "rice", "dal",
    "rajma chawal", "chana masala", "palak paneer", "paneer masala",
    "puri bhaji", "poha", "idli", "masala dosa",
    "sambhar", "uttapam", "medu vada", "samosa",
    "pav bhaji", "curd", "salad", "eggs"
]


def process_coco(input_path, output_path):
    print(f"\nProcessing: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Map original category_id → name
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # Keep only selected categories
    selected_ids = {
        cid for cid, name in id_to_name.items()
        if name in TARGET_CLASSES
    }

    print("Selected category IDs:", selected_ids)

    # New mapping: old_id → new_id (1..N)
    new_id_map = {}
    new_categories = []

    new_id = 1
    for cat in data['categories']:
        if cat['name'] in TARGET_CLASSES:
            new_id_map[cat['id']] = new_id
            new_categories.append({
                "id": new_id,
                "name": cat['name']
            })
            new_id += 1

    # Filter annotations
    new_annotations = []
    valid_image_ids = set()

    for ann in data['annotations']:
        if ann['category_id'] in selected_ids:
            new_ann = ann.copy()
            new_ann['category_id'] = new_id_map[ann['category_id']]
            new_annotations.append(new_ann)
            valid_image_ids.add(ann['image_id'])

    # Filter images (only those that still have annotations)
    new_images = [
        img for img in data['images']
        if img['id'] in valid_image_ids
    ]

    print(f"Images kept: {len(new_images)}")
    print(f"Annotations kept: {len(new_annotations)}")
    print(f"Categories kept: {len(new_categories)}")

    # Save new JSON
    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

    with open(output_path, 'w') as f:
        json.dump(new_data, f)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    base = "thali_detection/annotations"

    process_coco(
        os.path.join(base, "train/instances_train.json"),
        os.path.join(base, "train/instances_train_fixed.json")
    )

    process_coco(
        os.path.join(base, "val/instances_val.json"),
        os.path.join(base, "val/instances_val_fixed.json")
    )
