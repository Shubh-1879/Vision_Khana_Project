"""
Khana Dataset - Post-Training Analysis Script
Analyzes trained models and prepares leaderboard submission
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
from evaluate_model import evaluate_on_test_images

def load_best_models():
    """Load the best trained models"""
    models = {}

    # Try to load ResNet50 model first (preferred)
    resnet_path = os.path.expanduser('~/Vision_Khana_Project/best_model_resnet.pth')
    if os.path.exists(resnet_path):
        print("Loading ResNet50 model...")
        model = torchvision.models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 80)
        model.load_state_dict(torch.load(resnet_path, map_location='cpu'))
        models['resnet50'] = model
        print("✓ ResNet50 model loaded")
    else:
        print("✗ ResNet50 model not found")

    # Try to load custom CNN model
    custom_path = os.path.expanduser('~/Vision_Khana_Project/best_model.pth')
    if os.path.exists(custom_path):
        print("Loading Custom CNN model...")
        # Import the model architecture
        from khana_classification import SimpleKhanaCNN
        model = SimpleKhanaCNN(80)
        model.load_state_dict(torch.load(custom_path, map_location='cpu'))
        models['custom_cnn'] = model
        print("✓ Custom CNN model loaded")
    else:
        print("✗ Custom CNN model not found")

    return models

def analyze_training_logs():
    """Analyze training logs to extract final accuracies"""
    results = {}

    # Check ResNet50 logs
    resnet_log = os.path.expanduser('~/Vision_Khana_Project/training_output_resnet.log')
    if os.path.exists(resnet_log):
        with open(resnet_log, 'r') as f:
            content = f.read()
            # Extract final validation accuracy
            lines = content.split('\n')
            for line in reversed(lines):
                if 'Final Validation Accuracy:' in line:
                    acc = float(line.split(':')[1].strip().split('%')[0])
                    results['resnet50'] = {
                        'final_val_acc': acc,
                        'status': 'SUCCESS' if acc > 91 else 'BELOW_BASELINE'
                    }
                    break

    # Check Custom CNN logs
    custom_log = os.path.expanduser('~/Vision_Khana_Project/training_output.log')
    if os.path.exists(custom_log):
        with open(custom_log, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            for line in reversed(lines):
                if 'Final Validation Accuracy:' in line:
                    acc = float(line.split(':')[1].strip().split('%')[0])
                    results['custom_cnn'] = {
                        'final_val_acc': acc,
                        'status': 'SUCCESS' if acc > 91 else 'BELOW_BASELINE'
                    }
                    break

    return results

def prepare_leaderboard_submission(model_results, test_results):
    """Prepare submission format for leaderboard"""
    submission = {
        "team_name": "Vision_Khana_Project",
        "submission_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": "ResNet50_FineTuned",  # or "Custom_CNN"
        "validation_accuracy": 0.0,
        "test_predictions": []
    }

    # Use the best performing model
    best_model = max(model_results.keys(),
                    key=lambda x: model_results[x]['final_val_acc'])

    submission["model_used"] = best_model.replace('_', '_').title()
    submission["validation_accuracy"] = model_results[best_model]['final_val_acc']

    # Add test predictions
    for result in test_results:
        submission["test_predictions"].append({
            "image_path": os.path.basename(result["image_path"]),
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"]
        })

    return submission

def generate_report():
    """Generate comprehensive post-training report"""
    print("POST-TRAINING ANALYSIS REPORT")
    print("="*50)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Analyze training results
    training_results = analyze_training_logs()
    print("TRAINING RESULTS:")
    for model_name, results in training_results.items():
        print(f"  {model_name}:")
        print(".2f")
        print(f"    Status: {results['status']}")
        print()

    # Load models for evaluation
    models = load_best_models()
    if not models:
        print("❌ No trained models found!")
        return

    # Check for test images
    test_images_dir = os.path.expanduser('~/Vision_Khana_Project/test_images/')
    if os.path.exists(test_images_dir):
        test_images = [os.path.join(test_images_dir, f)
                      for f in os.listdir(test_images_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        test_images = []
        print("⚠️  No test images found in ~/Vision_Khana_Project/test_images/")
        print("   Please add 20-30 test images for evaluation")

    if test_images:
        print(f"Found {len(test_images)} test images")

        # Load class names
        data_path = os.path.expanduser('~/Vision_Khana_Project/dataset/khana/')
        if os.path.exists(data_path):
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(root=data_path)
            class_names = dataset.classes
        else:
            class_names = [f"class_{i}" for i in range(80)]

        # Evaluate best model
        best_model_name = max(training_results.keys(),
                            key=lambda x: training_results[x]['final_val_acc'])
        best_model = models[best_model_name]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_results = evaluate_on_test_images(best_model, test_images[:30], class_names, device)

        # Prepare leaderboard submission
        submission = prepare_leaderboard_submission(training_results, test_results)

        # Save submission
        submission_file = os.path.expanduser('~/Vision_Khana_Project/leaderboard_submission.json')
        with open(submission_file, 'w') as f:
            json.dump(submission, f, indent=2)

        print(f"\n✓ Leaderboard submission saved: {submission_file}")
        print(".2f")
        print(f"   Test predictions: {len(test_results)}")

    # Next steps
    print("\n" + "="*50)
    print("NEXT STEPS:")

    success_count = sum(1 for r in training_results.values() if r['status'] == 'SUCCESS')
    if success_count > 0:
        print("✅ Step 1 COMPLETE: Achieved >91% validation accuracy")
        print("   → Proceed to Step 2: Object Detection")
        print("   → Run: python setup_object_detection.py")
    else:
        print("❌ Step 1 INCOMPLETE: Below 91% baseline")
        print("   → Increase epochs, adjust learning rate, or try different architecture")
        print("   → Consider data augmentation or transfer learning")

    print("\n📊 Files generated:")
    print("   - leaderboard_submission.json (for submission)")
    print("   - evaluation_results.json (detailed test results)")
    print("   - project_status.json (updated project status)")

def main():
    generate_report()

if __name__ == "__main__":
    main()