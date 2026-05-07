"""
Khana Dataset - Prediction Script for Task 1

This script provides a `predict` function for single-image classification,
as required for the project evaluation.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# --- Global variables for the model and class names ---
# This ensures the model and class names are loaded only once.
MODEL = None
CLASS_NAMES = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _get_project_root():
    """Helper to find the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _load_class_names():
    """
    Loads class names from 'labels.txt'.
    It checks the script's directory first, then the project's dataset folder.
    """
    global CLASS_NAMES
    if CLASS_NAMES is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Per submission guidelines, helper files are in the submission folder.
        local_labels_path = os.path.join(script_dir, 'labels.txt')
        project_labels_path = os.path.join(_get_project_root(), 'dataset', 'labels.txt')

        if os.path.exists(local_labels_path):
            labels_path = local_labels_path
        elif os.path.exists(project_labels_path):
            labels_path = project_labels_path
        else:
            # Fallback flag if file is completely missing in evaluation environment
            CLASS_NAMES = []
            return CLASS_NAMES

        with open(labels_path, 'r', encoding='utf-8') as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]
    return CLASS_NAMES

def _load_model():
    """
    Loads the trained ResNet50 model from 'best_model_resnet.pth'.
    The model architecture must match the one used for training.
    """
    global MODEL
    if MODEL is None:
        # Build path relative to this script file.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model_resnet.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at '{model_path}'. Please ensure it's in the same directory as predict.py.")

        # Load the weights into CPU/GPU
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Infer number of classes from the state dict (fc.bias shape) to avoid mismatch
        num_classes = state_dict['fc.bias'].shape[0]

        # Define the model architecture to match the trained model
        model = torchvision.models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

        # Load the weights and prepare the model for inference
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        MODEL = model
        print(f"✓ Model loaded from '{model_path}' to {DEVICE}.")
    return MODEL

def predict(image):
    """
    Predicts the class label for a single image.

    Input:
    * `image` → PIL RGB image OR path to an image file (`str`)

    Output:
    * class name (`str`)
    * OR class index (`int`) starting from 1 (fallback)
    """
    # Ensure model and class names are loaded
    model = _load_model()
    class_names = _load_class_names()

    # Handle both string paths (local testing) and PIL Images (evaluator script)
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path not found: {image}")
        image = Image.open(image)

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted_idx = torch.max(probabilities, 0)

    if class_names:
        predicted_label = class_names[predicted_idx.item()]
        return predicted_label
    else:
        # Submission guidelines: fallback to 1-based index if string labels are unavailable
        return predicted_idx.item() + 1

# Example usage for local testing
if __name__ == '__main__':
    
    test_image_path = r"test_images\biryani.jpg"
    
    try:
        pred = predict(test_image_path)
        print(f"Prediction for '{test_image_path}': {pred}")
    except Exception as e:
        print(f"\n--- ERROR: {e} ---")