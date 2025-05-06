import cv2
import torch
from ultralytics import YOLO

"""
This file was used to test the model using the laptop camera, hasn't even been exported as .onnx.
"""

model = YOLO("runs/detect/train3/weights/best.pt")  

# model.export(format='onnx', half=True) # use for exporting

device = "mps" if torch.backends.mps.is_available() else "cpu"

cap = cv2.VideoCapture(1)  # 0 is my phone

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=device)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bound box
            conf = box.conf[0]  # confidence
            label = model.names[int(box.cls[0])]  # label

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # video
    cv2.imshow("YOLOv8 Webcam", frame)

    # exit = 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
