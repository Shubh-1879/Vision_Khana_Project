"""
Khana Dataset Classification - Evaluation Script
Evaluates trained model on 20-30 test images for leaderboard submission
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageFile
import warnings
from datetime import datetime

# Handle corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Palette images with Transparency")

def safe_pil_loader(path):
    """Safely load PIL image, handling corrupted files"""
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, IOError) as e:
        print(f"Warning: Could not load image {path}: {e}")
        return Image.new('RGB', (224, 224), color=(128, 128, 128))

def load_model(model_path, num_classes=80):
    """Load trained model"""
    model = torchvision.models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✓ Loaded model from: {model_path}")
    else:
        print(f"✗ Model not found: {model_path}")
        return None

    return model

def evaluate_on_test_images(model, test_image_paths, class_names, device='cpu'):
    """Evaluate model on individual test images"""
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []

    print(f"Evaluating {len(test_image_paths)} test images...")
    print("="*60)

    for img_path in test_image_paths:
        try:
            # Load and preprocess image
            img = safe_pil_loader(img_path)
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)

            predicted_label = class_names[predicted_class.item()]
            confidence_pct = confidence.item() * 100

            result = {
                'image_path': img_path,
                'predicted_class': predicted_label,
                'confidence': confidence_pct,
                'class_id': predicted_class.item()
            }

            results.append(result)

            print(f"Image: {os.path.basename(img_path)}")
            print(f"  Prediction: {predicted_label}")
            print(".2f")
            print("-"*40)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'image_path': img_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'class_id': -1
            })

    return results

def main():
    print("Khana Dataset - Model Evaluation Script")
    print("="*50)

    # Load class names from dataset
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(root_dir, 'dataset', 'khana')
    if os.path.exists(data_path):
        dataset = ImageFolder(root=data_path, transform=None, loader=safe_pil_loader)
        class_names = dataset.classes
        print(f"✓ Loaded {len(class_names)} class names")
    else:
        print("✗ Dataset not found, using generic class names")
        class_names = [f"class_{i}" for i in range(80)]

    # Load best model
    model_path = os.path.join(root_dir, 'step1_classification', 'best_model_resnet.pth')
    model = load_model(model_path, len(class_names))

    if model is None:
        print("Cannot proceed without model")
        return

    # TODO: Replace with actual test image paths
    # For now, create a placeholder - user needs to provide actual test images
    test_image_paths = [
        # Add paths to your 20-30 test images here
        # Example: "<repo_root>/step1_classification/test_images/image1.jpg"
    ]

    if not test_image_paths:
        print("\n" + "="*60)
        print("TEST IMAGE SETUP REQUIRED")
        print("="*60)
        print("Please add paths to your 20-30 test images in the test_image_paths list")
        print("Example:")
        print('test_image_paths = [')
        print(f'    "{os.path.join(root_dir, "step1_classification", "test_images", "aloo_gobi_001.jpg")}",')
        print(f'    "{os.path.join(root_dir, "step1_classification", "test_images", "chana_masala_002.jpg")}",')
        print('    # ... add more test images')
        print(']')
        print("\nOnce test images are ready, run this script again.")
        return

    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = evaluate_on_test_images(model, test_image_paths, class_names, device)

    # Save results
    import json
    results_file = os.path.join(root_dir, 'step1_classification', 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Calculate accuracy if ground truth is available
    # TODO: Add ground truth comparison when test labels are available

if __name__ == "__main__":
    main()
