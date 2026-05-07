import os
import torch
import cv2
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_class_names():
    labels_path = os.path.join(get_project_root(), 'dataset', 'labels.txt')
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_model(num_classes):
    try:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except:
        model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names()
    num_classes = len(class_names) + 1

    model = get_model(num_classes)

    model_path = "thali_detection/models/best_detection_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    input_dir = "thali_detection/images/test"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(image).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            if score < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            label_name = class_names[label-1]

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_cv, f"{label_name} {score:.2f}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1)

        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

    print("Inference complete. Check outputs/ folder.")

if __name__ == "__main__":
    main()