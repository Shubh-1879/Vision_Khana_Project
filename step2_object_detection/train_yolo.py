from ultralytics import YOLO

# 1. Load the pre-trained lightweight YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Train it for 50 epochs
results = model.train(data="yolo_dataset/data.yaml", epochs=50, imgsz=640)