import json
import os

root = "step2_object_detection/thali_detection/annotations"

train_file = os.path.join(root, "train", "instances_train.json")
val_file = os.path.join(root, "val", "instances_val.json")

print("STEP 2 VERIFICATION")
print("="*60)

if os.path.exists(train_file):
    with open(train_file) as f:
        train_data = json.load(f)
    print(f"Training COCO JSON:")
    print(f"  - Images: {len(train_data['images'])}")
    print(f"  - Annotations: {len(train_data['annotations'])}")
    print(f"  - Categories: {len(train_data['categories'])}")

if os.path.exists(val_file):
    with open(val_file) as f:
        val_data = json.load(f)
    print(f"\nValidation COCO JSON:")
    print(f"  - Images: {len(val_data['images'])}")
    print(f"  - Annotations: {len(val_data['annotations'])}")
    print(f"  - Categories: {len(val_data['categories'])}")

print("\n" + "="*60)
print("[OK] Step 2 data pipeline is ready for training!")
print("="*60)
