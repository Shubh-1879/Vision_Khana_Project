# Khana Dataset: Indian Cuisine Computer Vision Project

This repository contains a robust computer vision pipeline designed to classify and detect 80 different Indian food dishes from the Khana dataset. 

The project is structured into progressive steps, currently completing **Step 1 (Image Classification)** and **Step 2 (Object Detection)** using a highly optimized, two-stage YOLOv8 + ResNet50 architecture.

## 🗂️ Repository Structure

```text
├── dataset/
│   ├── labels.txt                 # The 80 Khana food classes
│   └── taxonomy.csv
├── step1_classification/          # Image Classification (Baseline > 91%)
│   ├── khana_classification_resnet.py
│   └── run_resnet.sh              # HPC submission script
├── step2_object_detection/        # Thali Object Detection
│   ├── train_yolo.py              # Fine-tunes YOLOv8n on custom thali items
│   ├── task2_yolo_classifier.py   # Full pipeline: Detects -> Crops -> Classifies
│   ├── yolo_dataset/              # YOLO formatted training data
│   └── outputs/                   # Final bounded images
└── README.md
