import cv2
from picamera2 import Picamera2
import onnxruntime as ort
import numpy as np
import time

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # might be able to remove
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img


def format_opencv(x):
    #  format: (center_x, center_y, w, h) -> (x1, y1, x2, y2)
    y = x.copy()
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    
    return map(int, y)

print("Loading Model...")
session = ort.InferenceSession("./best.onnx")
input_name = session.get_inputs()[0].name

print("Starting Capture...")
cap = Picamera2()
config = cap.create_preview_configuration({"format":"RGB888", "size":(320, 320)})
cap.configure(config)
cap.start()

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()

first_des = None


print("Starting Loop...")
print("\n")
saved = 3
while True:
    start = time.time()
    
    frame = cap.capture_array()
    if frame is None or frame.size == 0:
        print("Missing Frame, exiting loop")
        break

    input_image = preprocess(frame)
    outputs = session.run(None, {input_name: input_image})[0]  
    outputs = np.squeeze(outputs)

    if len(outputs.shape) == 1:
        outputs = np.expand_dims(outputs, axis=0)

    best_conf = 0
    best_crop = None
    best_coords = None
    
    count = 0
    for det in outputs.T:
        if det[4] < 0.7:
            continue
        # print(det.shape, "|", det)
        x1, y1, x2, y2 = format_opencv(det[:4])
        count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    end = time.time()
    if saved > 0 and count > 1:
        cv2.imwrite(f"saved[{saved}].jpg", frame)
        saved -= 1
        
    # ~ cv2.imshow("ONNX", frame)
    
    print(f"\033[A\r{1/(end-start):f} fps, {count} detected")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Stoping Capture...")
cap.stop()
cv2.destroyAllWindows()

print("Script Done")
