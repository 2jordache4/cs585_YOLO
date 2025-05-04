import cv2
import torch
from ultralytics import YOLO

print("Loading Model ...")
model = YOLO("./best.onnx")  # use onnx model
print("Model Loaded\n")
# model.export(format='onnx', half=True) # use for exporting

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Opening Camera ...")
cap = cv2.VideoCapture(0)  # 0 is my phone

if cap.isOpened():
    print("Camera Opened\nStarting Loop ...\n")

while cap.isOpened():
    # capture frame
    ret, frame = cap.read()
    
    # check if frame was read correctly
    if not ret:
        break
    
    # reformat frame
    frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
    print("Input Shape:",frame.shape)

    # predict on frame
    results = model(frame, device=device)
    
    # plot bounding box
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bound box
            conf = box.conf[0]  # confidence
            label = model.names[int(box.cls[0])]  # label

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # display frame
    cv2.imshow("YOLOv8 Webcam", frame)

    # exit = 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Script Done")