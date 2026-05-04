# Vision_Khana_Project

This repository is organized into two separate parts:

- `step1_classification/`: ResNet50-based food classification for 80 Khana classes.
- `step2_object_detection/`: Thali object detection setup and training using Faster R-CNN.
- `shared/`: Common utilities and status tracking.

## Directory structure

```
Vision_Khana_Project/
├── README.md
├── dataset/                    # Classification dataset
├── pretrained_weights/         # Local pretrained model weights
├── shared/                    # Shared scripts and status files
│   ├── download_weights.py
│   ├── project_status.py
│   └── rename.py
├── step1_classification/      # ResNet50 classification step
│   ├── khana_classification_resnet.py
│   ├── analyze_results.py
│   ├── evaluate_model.py
│   ├── run_resnet.sh
│   ├── setup_test_images.py
│   └── test_images/
└── step2_object_detection/    # Thali object detection step
    ├── annotation_helper.py
    ├── setup_object_detection.py
    ├── train_object_detection.py
    ├── test_object_detection.py
    ├── run_detection_training.sh
    └── thali_detection/
```
