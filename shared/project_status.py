"""
Khana Dataset - Project Status Tracker
Tracks progress across all 4 steps of the project
"""

import os
import json
from datetime import datetime

def get_project_status():
    """Get current status of all project components"""

    status = {
        "project_name": "Khana Dataset Classification & Detection",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {
            "step1_classification": {
                "name": "Image Classification (80 classes)",
                "target_accuracy": ">91% validation accuracy",
                "status": "COMPLETED",
                "models": {
                    "resnet50_finetune": {
                        "script": "step1_classification/khana_classification_resnet.py",
                        "job_script": "step1_classification/run_resnet.sh",
                        "logs": ["step1_classification/training_output_resnet.log", "step1_classification/training_error_resnet.log"],
                        "model_file": "step1_classification/best_model_resnet.pth",
                        "status": "SUCCESS",
                        "validation_accuracy": 91.15
                    }
                },
                "dataset": {
                    "location": "~/Vision_Khana_Project/dataset/khana/",
                    "classes": 80,
                    "total_images": 131819,
                    "train_split": "80%",
                    "val_split": "20%"
                }
            },
            "step2_object_detection": {
                "name": "Object Detection on Clean Thali Images",
                "status": "SETUP_READY",
                "requirements": [
                    "Clean thali images (no background clutter)",
                    "COCO-format bounding box annotations",
                    "80 food class labels"
                ],
                "setup_script": "step2_object_detection/setup_object_detection.py",
                "dataset_structure": "~/Vision_Khana_Project/step2_object_detection/thali_detection/",
                "model": "Faster R-CNN with ResNet50 backbone",
                "next_steps": [
                    "Collect/prepare thali images",
                    "Create bounding box annotations",
                    "Train detection model"
                ]
            },
            "step3_bev_transformation": {
                "name": "BEV Transformation for Natural Images",
                "status": "NOT_STARTED",
                "requirements": [
                    "Natural scene images with food items",
                    "Camera calibration parameters",
                    "3D to 2D transformation logic"
                ],
                "dependencies": ["Step 2 object detection model"]
            },
            "step4_nrf_creation": {
                "name": "NRF Creation (Bonus)",
                "status": "NOT_STARTED",
                "requirements": [
                    "Step 1-3 completion",
                    "Nutritional database",
                    "Recipe analysis"
                ]
            }
        },
        "infrastructure": {
            "local_machine": {
                "os": "Windows",
                "gpu": "Unknown",
                "development": "VS Code + Git"
            },
            "hpc_cluster": {
                "address": "10.1.4.95",
                "user": "shubham.agarwal_phd24",
                "gpu": "Tesla V100-PCIE-32GB",
                "queue": "gpu",
                "data_location": "~/Vision_Khana_Project/"
            },
            "version_control": {
                "platform": "GitHub",
                "repo": "Vision_Khana_Project",
                "sync_method": "Git + rsync"
            }
        },
        "deliverables": {
            "step1": [
                "Trained classifier >91% accuracy",
                "Evaluation on 20-30 test images",
                "Leaderboard submission"
            ],
            "step2": [
                "Object detection model for thali images",
                "Detection accuracy metrics"
            ],
            "step3": [
                "BEV transformation pipeline",
                "Natural image detection results"
            ],
            "step4": [
                "NRF generation system",
                "Nutritional analysis reports"
            ]
        }
    }

    return status

def save_status_report():
    """Save current project status to JSON file"""
    status = get_project_status()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    status_file = os.path.join(root_dir, 'project_status.json')
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    print(f"✓ Project status saved to: {status_file}")
    return status

def print_status_summary():
    """Print a concise status summary"""
    status = get_project_status()

    print("KHANA PROJECT STATUS SUMMARY")
    print("="*50)
    print(f"Last Updated: {status['last_updated']}")
    print()

    for step_key, step_info in status['steps'].items():
        step_num = step_key.split('_')[0].replace('step', '')
        print(f"Step {step_num}: {step_info['name']}")
        print(f"  Status: {step_info['status']}")

        if 'models' in step_info:
            for model_name, model_info in step_info['models'].items():
                print(f"  {model_name}: {model_info['status']}")

        if step_info['status'] == 'SETUP_READY' and 'next_steps' in step_info:
            print("  Next: " + step_info['next_steps'][0])

        print()

    print("INFRASTRUCTURE:")
    print(f"  Local: {status['infrastructure']['local_machine']['os']}")
    print(f"  HPC: {status['infrastructure']['hpc_cluster']['gpu']} GPU")
    print(f"  GitHub: {status['infrastructure']['version_control']['repo']}")

def main():
    print_status_summary()
    save_status_report()

if __name__ == "__main__":
    main()