import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")
device = "mps" 
cap = cv2.VideoCapture(0)

first_des = None
sift = cv2.SIFT_create()
matcher = cv2.BFMatcher() #brute force matcher

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 320))
    results = model(frame, device=device)
    best_conf = 0
    best_crop = None
    best_coords = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            label = model.names[int(box.cls[0])]

            if conf > best_conf:
                best_conf = conf
                best_coords = (x1, y1, x2, y2)
                best_crop = frame[y1:y2, x1:x2]

    if best_crop is not None:
        gray_crop = cv2.cvtColor(best_crop, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray_crop, None)

        if first_des is None and des is not None:
            first_des = des
            # save the first descriptors 
            # I kinda want to make this like adapting so it can save features specific to a person
            # rather than person + background but right now i think this is okay

        elif des is not None and first_des is not None:
            matches = matcher.knnMatch(first_des, des, k=2)
            good = []
            for m, n in matches: 
              if m.distance < 0.8 * n.distance: 
                  good.append([m]) 

            if len(good) > 20:  # kp threshhold, 10 was pretty consistent but im not convinced its enough
                x1, y1, x2, y2 = best_coords
                cv2.putText(frame, f"PERSON 1 {conf:.2f}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # not the same person
                x1, y1, x2, y2 = best_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("test 2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
