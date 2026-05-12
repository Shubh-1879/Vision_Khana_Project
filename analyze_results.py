"""
Khana Dataset - Post-Training Analysis Script
Analyzes trained models and prepares leaderboard submission
"""

import torch
import torchvision
import os
import json
from datetime import datetime

# Since this file is now in the same folder as evaluate_model.py, we can import it directly!
from evaluate_model import evaluate_on_test_images

def get_project_root():
    """Helper to find the project root directory (one folder up)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_best_models():
    """Load the best trained ResNet50 model"""
    models = {}
    root_dir = get_project_root()

    # Load ResNet50 model
    resnet_path = os.path.join(root_dir, 'step1_classification', 'best_model_resnet.pth')
    if os.path.exists(resnet_path):
        print("Loading ResNet50 model...")
        # Recreate ResNet50 architecture and load state_dict
        model = torchvision.models.resnet50(weights=None)
        # Freeze early layers (same as training)
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        # Replace classifier for 80 classes
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 80)
        # Load saved weights
        state_dict = torch.load(resnet_path, map_location='cpu')
        model.load_state_dict(state_dict)
        models['resnet50'] = model
        print("✓ ResNet50 model loaded")
    else:
        print(f"✗ ResNet50 model not found at: {resnet_path}")

    return models

def analyze_training_logs():
    """Analyze training logs to extract final accuracies"""
    results = {}
    root_dir = get_project_root()

    # Check ResNet50 logs
    resnet_log = os.path.join(root_dir, 'step1_classification', 'training_output_resnet.log')
    if os.path.exists(resnet_log):
        with open(resnet_log, 'r') as f:
            content = f.read()
            # Extract final validation accuracy
            lines = content.split('\n')
            for line in reversed(lines):
                if 'Final Validation Accuracy:' in line:
                    acc = float(line.split(':')[1].strip().replace('%',''))
                    results['resnet50'] = {
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
        "model_used": "ResNet50_FineTuned",
        "validation_accuracy": 0.0,
        "test_predictions": []
    }

    # Use ResNet50 results
    if 'resnet50' in model_results:
        submission["validation_accuracy"] = model_results['resnet50']['final_val_acc']

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
        print(f"    Final Validation Accuracy: {results['final_val_acc']:.2f}%")
        print(f"    Status: {results['status']}")
        print()

    # Load models for evaluation
    models = load_best_models()
    if not models:
        print("❌ No trained models found!")
        return

    root_dir = get_project_root()
    # Check for test images
    test_images_dir = os.path.join(root_dir, 'step1_classification', 'test_images')
    if os.path.exists(test_images_dir):
        test_images = [os.path.join(test_images_dir, f)
                      for f in os.listdir(test_images_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        test_images = []
        print(f"⚠️  No test images found in {test_images_dir}")
        print("   Please add 20-30 test images for evaluation")

    if test_images:
        print(f"Found {len(test_images)} test images")

        # Load class names
        data_path = os.path.join(root_dir, 'dataset', 'khana')
        if os.path.exists(data_path):
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(root=data_path)
            class_names = dataset.classes
        else:
            class_names = [f"class_{i}" for i in range(80)]

        # Evaluate ResNet50 model
        best_model = models['resnet50']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_results = evaluate_on_test_images(best_model, test_images[:30], class_names, device)

        # Prepare leaderboard submission
        submission = prepare_leaderboard_submission(training_results, test_results)

        # Save submission to the main project folder
        submission_file = os.path.join(root_dir, 'leaderboard_submission.json')
        with open(submission_file, 'w') as f:
            json.dump(submission, f, indent=2)

        print(f"\n✓ Leaderboard submission saved: {submission_file}")
        print(f"   Validation Accuracy in Report: {submission['validation_accuracy']:.2f}%")
        print(f"   Test predictions: {len(test_results)}")

    # Next steps
    print("\n" + "="*50)
    print("NEXT STEPS:")

    success_count = sum(1 for r in training_results.values() if r['status'] == 'SUCCESS')
    if success_count > 0:
        print("✅ Step 1 COMPLETE: Achieved >91% validation accuracy")
        print("   -> Proceed to Step 2: Object Detection")
    else:
        print("❌ Step 1 INCOMPLETE: Below 91% baseline")
        print("   -> Increase epochs, adjust learning rate, or try different architecture")

def main():
    generate_report()

if __name__ == "__main__":
    main()
