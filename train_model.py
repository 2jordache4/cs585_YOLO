import os
from ultralytics import YOLO
import torch

"""
Python script used to train the models
"""

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# yolo nano model (smallest for raspi)
model = YOLO('yolov8n.pt')


# Train YOLOv8 on the COCO Person Dataset
results = model.train(
    data='./cs585_YOLO/data.yaml',
    epochs=3,
    batch=2,
    imgsz=320,
    seed=32,
    cache=False  
) # Test run 1

# results = model.train(
#     data='./cs585_YOLO/data.yaml',
#     epochs=3,
#     batch=2,
#     imgsz=320,
#     seed=32,
#     cache=False,
#     device=device,
#     label_smoothing=0.1,
#     cos_lr=True
# )  # Test run 2

# results = model.train(
#     data='./cs585_YOLO/data_grey.yaml',
#     epochs=3,
#     batch=2,
#     imgsz=320,
#     seed=32,
#     cache=False  
# ) #gray

