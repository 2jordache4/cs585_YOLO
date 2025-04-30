import cv2
import onnxruntime as ort
import numpy as np
import time

device_is_pi = False

if device_is_pi:
    from picamera2 import Picamera2
    
    def init_camera():
        cap = Picamera2()
        config = cap.create_preview_configuration({"format":"RGB888", "size":(320, 320)})
        cap.configure(config)
        cap.start()
    
        return cap

    def close_camera(cap):
        cap.stop()

    def get_frame(cap):
        frame = cap.capture_array()
        
        in_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # might be able to remove
        in_img = in_img.astype(np.float32) / 255.0
        in_img = np.transpose(in_img, (2, 0, 1))  # HWC → CHW
        in_img = np.expand_dims(in_img, axis=0)  # Add batch dim
        
        return frame, in_img
    
    def get_frame_grey(cap):
        frame = cap.capture_array()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gray_3ch = cv2.merge([gray, gray, gray])               # (H, W, 3) → replicate channels
        in_img = gray_3ch.astype(np.float32) / 255.0
        in_img = np.transpose(in_img, (2, 0, 1))               # (3, H, W)
        in_img = np.expand_dims(in_img, axis=0) 
        
        return gray, in_img
        
else:
    def init_camera():
        cap = cv2.VideoCapture(0)
    
        return cap

    def close_camera(cap):
        cap.release()

    def get_frame_grey(cap):
        ret, frame = cap.read()
        
        if not ret:
            quit()
        
        frame = cv2.resize(frame, (320, 320))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gray_3ch = cv2.merge([gray, gray, gray])               # (H, W, 3) → replicate channels
        in_img = gray_3ch.astype(np.float32) / 255.0
        in_img = np.transpose(in_img, (2, 0, 1))               # (3, H, W)
        in_img = np.expand_dims(in_img, axis=0) 
        # CHANGED THESE AS WELL
        return gray, in_img
    
    def get_frame(cap):
        ret, frame = cap.read()
        
        if not ret:
            quit()
        
        frame = cv2.resize(frame, (320, 320))
        
        in_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # might be able to remove
        in_img = in_img.astype(np.float32) / 255.0
        in_img = np.transpose(in_img, (2, 0, 1))  # HWC → CHW
        in_img = np.expand_dims(in_img, axis=0)  # Add batch dim
        return frame, in_img


class CarUnit:
    def __init__(self, dev, ultra_pin=None):
        self.dev = dev
        self.ultra = ultra_pin
        
    def connect(self):
        # open dev/file
        pass
        
    def release(self):
        # close dev/file
        pass
    
    def write(r_motor=0, l_motor=0):
        # encode motor values
        
        # send over serial
        pass
    
    def set_ultra(self, thresh):
        # build config msg
        
        # sent over serial
        pass

class SIFTdetect:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        self.target = None
        
    def match_target(self, frame, bboxes):
        # not boxes availible
        if bboxes.shape[0] == 0:
            return None
        
        # get best box index
        best_idx = np.argmax(bboxes[:,4])
        best_box = bboxes[best_idx, :]
        
        # crop the best box area
        if best_box[0] < 0: best_box[0] = 0
        if best_box[1] < 0: best_box[1] = 0
        if best_box[2] > frame.shape[0]: best_box[2] = frame.shape[0]
        if best_box[3] > frame.shape[0]: best_box[3] = frame.shape[0]
        
        best_crop = cv2.cvtColor(frame[best_box[1]:best_box[3], best_box[0]:best_box[2]], cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(best_crop, None)
        
        if self.target is None:
            self.target = des
        
        else:
            matches = self.matcher.knnMatch(self.target, des, k=2)
            
            good = []  
            for m, n in matches: 
                if m.distance < 0.8 * n.distance: 
                    good.append([m])
                  
            if len(good) > 20: # kp threshhold, 10 was pretty consistent but im not convinced its enough
                self.target = des # re-update the descriptorsq
                return best_idx
            
    def match_target_grey(self, frame, bboxes):
        # not boxes availible
        if bboxes.shape[0] == 0:
            return None
        
        # get best box index
        best_idx = np.argmax(bboxes[:,4])
        best_box = bboxes[best_idx, :]
        
        # crop the best box area
        if best_box[0] < 0: best_box[0] = 0
        if best_box[1] < 0: best_box[1] = 0
        if best_box[2] > frame.shape[0]: best_box[2] = frame.shape[0]
        if best_box[3] > frame.shape[0]: best_box[3] = frame.shape[0]
        
        best_crop = frame[best_box[1]:best_box[3], best_box[0]:best_box[2]]
        kp, des = self.sift.detectAndCompute(best_crop, None)
        
        if self.target is None:
            self.target = des
        
        else:
            matches = self.matcher.knnMatch(self.target, des, k=2)
            
            good = []  
            for m, n in matches: 
                if m.distance < 0.8 * n.distance: 
                    good.append([m])
                  
            if len(good) > 20: # kp threshhold, 10 was pretty consistent but im not convinced its enough
                self.target = des
                return best_idx

def process_output(results, thresh, tol, bbox=False):
    
    results = np.squeeze(results)

    if len(results.shape) == 1:
        results = np.expand_dims(results, axis=0)
    
    # loops through each result, 
    #   throws out any confidance below threshold,
    #   and removes any duplicates
    output_list = []
    output = np.array(output_list)   
    for result in results.T:
        if result[4] < thresh: continue
        
        if output_list:
            diff = (output + result) / (2*result)
            
            if np.all(diff > tol): continue
            
        output_list.append(result)
        output = np.array(output_list)
    
    output = output.astype(int)
    
    if bbox:       
        #  format: (center_x, center_y, w, h) -> (x1, y1, x2, y2)
        
        if output.shape[0] == 0:
            return output, np.zeros((0,))

        bbox_out = np.zeros(output.shape)

        bbox_out[:,0] = output[:,0] - output[:,2]/2
        bbox_out[:,1] = output[:,1] - output[:,3]/2
        bbox_out[:,2] = output[:,0] + output[:,2]/2
        bbox_out[:,3] = output[:,1] + output[:,3]/2
        bbox_out[:,4] = output[:,4]
        
        bbox_out = bbox_out.astype(int)
        
        return output, bbox_out
    
    else:
        return output

def add_bboxes(frame, bboxes, target=None):
    
    if bboxes.shape[0] == 0:
        return
    
    if len(bboxes.shape) == 1:
        bboxes= np.expand_dims(bboxes, axis=0)
    
    for bbox in bboxes:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
    if target is not None:
        x_val = int((bboxes[target, 0] + bboxes[target,2]) / 2)
        cv2.line(frame, (x_val, 0), (x_val, frame.shape[0]), (255, 0, 0), 2)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # might be able to remove
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # add batch dim
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
session1 = ort.InferenceSession("./runs/detect/train3/weights/best.onnx")
session2 = ort.InferenceSession("./runs/detect/train5/weights/grey.onnx")
input_name1 = session1.get_inputs()[0].name
input_name2 = session2.get_inputs()[0].name

print("Starting Capture...")
cap = init_camera()

sifter = SIFTdetect()
sifter_grey = SIFTdetect()

first_des = None

print("Starting Loop...")
print("\n")
saved = 3
f = open("color_fps.txt", 'w')
f2 = open("grey_fps.txt", 'w')
timer_start = time.time()
timer_end = timer_start + 10

while True:
    start1 = time.time()
    timer_start = time.time()
    
    # get camera image
    frame, in_img = get_frame(cap)
    
    # run image through model
    outputs1 = session1.run(None, {input_name1: in_img})[0]   
    
    # process model output
    outputs1, bboxes1 = process_output(outputs1, 0.7, 0.8, bbox=True)
    
    # apply sift
    target_idx = sifter.match_target(frame, bboxes1)
    
    # put boxes on image
    add_bboxes(frame, bboxes1, target_idx)
    end1 = time.time()


    # Grey Mode B)
    start2 = time.time()

    frame_grey, in_img_grey = get_frame_grey(cap)
    outputs2 = session2.run(None, {input_name1: in_img_grey})[0] 
    outputs2, bboxes2 = process_output(outputs2, 0.7, 0.8, bbox=True)
    target_idx2 = sifter_grey.match_target_grey(frame_grey, bboxes2)
    add_bboxes(frame_grey, bboxes2, target_idx2)

    end2 = time.time()

    print(f"\033[A\r{1/(end1-start1):f} color fps, {1/(end2-start2):f} grey fps, {outputs1.shape[0]} color detected, {outputs2.shape[0]} grey detected")

    if(timer_start < timer_end):
        f.write(str(1/(end1-start1)))
        f.write("\n")
        f2.write(str(1/(end2-start2)))
        f2.write("\n")
    else:
        f.close()
        f2.close()
        print("Done")
    
    if saved > 0 and outputs1.shape[0] > 1:
        cv2.imwrite(f"saved[{saved}].jpg", frame)
        saved -= 1
        
    cv2.imshow("ONNX", frame)
    cv2.imshow("ONNX FULL", frame_grey)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Stoping Capture...")
close_camera(cap)
cv2.destroyAllWindows()

print("Script Done")
