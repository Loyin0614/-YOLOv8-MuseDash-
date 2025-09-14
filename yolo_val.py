# This file is for YOLO varification
from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train8/weights/best.pt')

# Use the model
if __name__ == '__main__':
    metrics = model.val(data='peropero.yaml', batch=16)  # evaluate model performance on the validation set