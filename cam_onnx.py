import cv2
import numpy as np
import onnxruntime as ort
import time

session = ort.InferenceSession("./best.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()

first_des = None


def preprocess(img):
    frame = cv2.resize(img, (320, 320))
    
    img = cv2.resize(img, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img, frame


def format_opencv(x):
    #  format: (center_x, center_y, w, h) -> (x1, y1, x2, y2)
    y = x.copy()
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    
    return map(int, y)

first = True
while cap.isOpened():
    start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break

    input_image, frame = preprocess(frame)
    outputs = session.run(None,
                          {input_name: input_image})[0]  

    outputs = np.squeeze(outputs)

    if len(outputs.shape) == 1:
        outputs = np.expand_dims(outputs, axis=0)

    best_conf = 0
    best_crop = None
    best_coords = None
        
    for det in outputs.T:
        if det[4] < 0.7:
            continue
        # print(det.shape, "|", det)
        x1, y1, x2, y2 = format_opencv(det[:4])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    end = time.time()
    cv2.imshow("ONNX", frame)
    
    if first:
        first = False
        print("\n")
    
    print(f"\033[A\r{1/(end-start):f} fps")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
