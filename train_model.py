import os
from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# yolo nano model (smallest for raspi)
model = YOLO('yolov8n.pt')

#model.export(format="onnx")
#model.export(format="onnx", half=True)
# exporting as an onnx with half precision will theoretically make it run faster on a rasp pi3
# According to chatgpt the following is true:
# Faster Training → Lower Accuracy
# Less RAM Usage → Slightly Less Detail
# Faster Pi Inference → Smaller Model

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
#     data='./cs585_YOLO/data.yaml',
#     epochs=1,  # was 150, went to 75
#     batch=1,  # for my mac ram
#     imgsz=256,  # was 640, dropped to 320
#     seed=32,  # reproduceability 
#     optimizer='NAdam',  # Adaptive learning optimizer
#     weight_decay=1e-4,  # Regularization to prevent overfitting
#     momentum=0.937,  # Initial momentum
#     cos_lr=True,  # Cosine learning rate decay
#     lr0=0.01,  # Initial learning rate
#     lrf=1e-5,  # Final learning rate
#     warmup_epochs=2,  # was 10, went to 5 bc epochs was lowered
#     warmup_momentum=0.5,  # Adjusted warmup momentum
#     close_mosaic=20,  # Helps with augmentation stability 
#     dropout=0.5,  # Prevents overfitting
#     verbose=True,  # Prints training progress
#     device=device  # Ensure correct device usage
# )



# results = model.train(
#     data='./cs585_YOLO/data.yaml',
#     epochs=75,  # was 150, went to 75
#     batch_size=2,  # for my mac ram
#     imgsz=320,  # was 640, dropped to 320
#     seed=32,  # reproduceability 
#     optimizer='NAdam',  # Adaptive learning optimizer
#     weight_decay=1e-4,  # Regularization to prevent overfitting
#     momentum=0.937,  # Initial momentum
#     cos_lr=True,  # Cosine learning rate decay
#     lr0=0.01,  # Initial learning rate
#     lrf=1e-5,  # Final learning rate
#     warmup_epochs=5,  # was 10, went to 5 bc epochs was lowered
#     warmup_momentum=0.5,  # Adjusted warmup momentum
#     close_mosaic=20,  # Helps with augmentation stability
#     label_smoothing=0.2,  # Prevents overconfidence in predictions
#     dropout=0.5,  # Prevents overfitting
#     verbose=True,  # Prints training progress
#     device=device,  # Ensure correct device usage
#     cache=True  # apparently will make it less hard on ram
# )
# a lot of this was taken from
# https://github.com/J3lly-Been/YOLOv8-HumanDetection/blob/main/training.ipynb
