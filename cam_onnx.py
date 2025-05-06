import cv2
import numpy as np
import onnxruntime as ort
"""
This file was used to first test ONNX and ensure that it would properly run after being exported
"""

session = ort.InferenceSession("runs/detect/train3/weights/best.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()

first_des = None


def preprocess(img):
    img = cv2.resize(img, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img


def format_opencv(x):
    #  format: (center_x, center_y, w, h) -> (x1, y1, x2, y2)
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = preprocess(frame)
    outputs = session.run(None,
                          {input_name: input_image})[0]  
    outputs = np.squeeze(outputs)

    if len(outputs.shape) == 1:
        outputs = np.expand_dims(outputs, axis=0)

    best_conf = 0
    best_crop = None
    best_coords = None

    for det in outputs:
        if det[4] < 0.5:
            continue
        class_id = int(det[5])

        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])

        if conf > best_conf:
            best_conf = conf
            best_coords = (x1, y1, x2, y2)
            best_crop = frame[y1:y2, x1:x2]

    if best_crop is not None:
        gray_crop = cv2.cvtColor(best_crop, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray_crop, None)

        if first_des is None and des is not None:
            first_des = des

        elif des is not None and first_des is not None:
            matches = matcher.knnMatch(first_des, des, k=2)
            for m, n in matches: 
              if m.distance < 0.8 * n.distance: 
                  good.append([m]) 

            if len(good) > 5:
                x1, y1, x2, y2 = best_coords
                cv2.putText(frame, f"PERSON 1 {best_conf:.2f}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                x1, y1, x2, y2 = best_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("ONNX", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
