"""
Task 2: Thali Detection & Classification
Uses custom fine-tuned YOLOv8 for bounding box detection 
and Task 1 ResNet50 for classification.
Generates bounded images and a predictions JSON file.
"""

import os
import cv2
import json
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# --- CONFIGURATION PATHS ---
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt" 
RESNET_MODEL_PATH = "../step1_classification/best_model_resnet.pth"
LABELS_PATH = "../dataset/labels.txt"
INPUT_DIR = "yolo_dataset/images/test"
OUTPUT_DIR = "outputs"

def load_class_names(labels_path):
    """Loads the 80 food classes directly from the dataset labels file."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Error: {labels_path} not found. Please check your path!")
        
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_resnet_classifier(model_path, num_classes, device):
    """Loads the fine-tuned ResNet50 from Task 1."""
    model = torchvision.models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Loaded ResNet50 from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"ResNet50 model not found at {model_path}.")

def main():
    print("\n" + "="*50)
    print("Starting Task 2 Pipeline: YOLO + ResNet50")
    print("="*50 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Classes dynamically from labels.txt
    class_names = load_class_names(LABELS_PATH)
    print(f"✓ Loaded {len(class_names)} classes from {LABELS_PATH}")
    
    # 2. Load ResNet50
    classifier = load_resnet_classifier(RESNET_MODEL_PATH, len(class_names), device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load YOLO
    print(f"Loading custom YOLO model from {YOLO_MODEL_PATH}...")
    detector = YOLO(YOLO_MODEL_PATH)

    # 4. Process Images
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nFound {len(image_files)} images to process.\n")

    if len(image_files) == 0:
        print("No images found in the input directory. Exiting.")
        return

    # Dictionary to hold the final predictions for the JSON file
    ta_predictions_dict = {}

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        img_cv2 = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Get the base name without extension (e.g. "Plate 67.jpg" -> "Plate 67")
        base_name = os.path.splitext(img_name)[0]
        current_image_predictions = []

        results = detector(img_rgb, conf=0.25, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        print(f"Processing {img_name}: Found {len(boxes)} food items.")

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            input_tensor = transform(crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, predicted_class = torch.max(probabilities, 0)
                
            pred_label = class_names[predicted_class.item()]
            pred_conf = conf.item()
            
            current_image_predictions.append(pred_label)

            # Draw on the image
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{pred_label} ({pred_conf:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_cv2, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(img_cv2, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add this image's predictions to the main dictionary
        ta_predictions_dict[base_name] = current_image_predictions

        # Save the final drawn image
        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, img_cv2)
        print(f"  -> Saved bounded image to: {out_path}")

    # Save the JSON file
    json_output_path = "task2_predictions.json"
    with open(json_output_path, 'w') as json_file:
        json.dump(ta_predictions_dict, json_file, indent=2)
        
    print("\n" + "="*50)
    print(f"Task 2 Complete!")
    print(f"1. Bounded images saved in: '{OUTPUT_DIR}' folder.")
    print(f"2. TA Predictions JSON saved as: '{json_output_path}'")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()