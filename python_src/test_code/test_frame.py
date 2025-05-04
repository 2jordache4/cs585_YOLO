import onnxruntime as ort
import numpy as np
import time

# CarUnit, SIFTDetect, and TargetUnit
from modlib import *

import cv2

def init_camera():
    cap = cv2.VideoCapture(0)

    return cap

def close_camera(cap):
    cap.release()

def get_frame(cap):
    ret, frame = cap.read()
    
    if not ret:
        quit()
    
    frame = cv2.resize(frame, (320, 320))
    
    
    return frame
    
def pre_proc_rgb(frame):
    frame = cv2.resize(frame, (320, 320))
    in_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # might be able to remove
    in_img = in_img.astype(np.float32) / 255.0
    in_img = np.transpose(in_img, (2, 0, 1))  # HWC → CHW
    in_img = np.expand_dims(in_img, axis=0)  # Add batch dim
    
    return frame, in_img
    
def pre_proc_gray(frame):
    frame = cv2.resize(frame, (320, 320))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # might be able to remove
    frame = cv2.merge((frame, frame, frame))
    in_img = frame.astype(np.float32) / 255.0
    in_img = np.transpose(in_img, (2, 0, 1))  # HWC → CHW
    in_img = np.expand_dims(in_img, axis=0)  # Add batch dim
    
    return frame, in_img

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

cap = init_camera()

print("Loading Model...")
session_rgb = ort.InferenceSession("./best.onnx")
input_name_rgb = session_rgb.get_inputs()[0].name

session_gray = ort.InferenceSession("./grey.onnx")
input_name_gray = session_gray.get_inputs()[0].name

sifter = SIFTdetect()

print("Starting Loop...")
print("\n")

first = True

while True:
    # timing for fps
    start_cap = time.time()
    
    # get camera image
    ret, frame = cap.read()
    
    if not ret:
        quit()
    
    end_cap = time.time()
    
    ######################### RGB ################################
    start_rgb = time.time()
    
    # process through model
    frame_rgb, in_img = pre_proc_rgb(frame)
    outputs = session_rgb.run(None, {input_name_rgb: in_img})[0]  
    
    # process model output
    outputs, bboxes = process_output(outputs, thresh=0.7, tol=0.6, bbox=True)
    
    # apply sift
    target_idx = sifter.match_target(frame_rgb, bboxes)

    end_rgb= time.time()
    
    # put boxes on image
    add_bboxes(frame_rgb, bboxes, target_idx)
        
    cv2.imshow("ONNX_rgb", frame_rgb)
    
    ######################### RGB ################################
    start_gray = time.time()
    
    # process through model
    frame_gray, in_img = pre_proc_gray(frame)
    outputs = session_gray.run(None, {input_name_gray: in_img})[0]  
    
    # process model output
    outputs, bboxes = process_output(outputs, thresh=0.7, tol=0.6, bbox=True)
    
    # apply sift
    target_idx = sifter.match_target(frame_gray, bboxes)

    end_gray = time.time()
    
    # put boxes on image
    add_bboxes(frame_gray, bboxes, target_idx)
        
    cv2.imshow("ONNX_gray", frame_gray)
    
    if first:
        first = False
        print("\n")
    
    print(f"\033[A\r\t\t\tColor: {1/(end_cap-start_cap + end_rgb-start_rgb):.3f} fps  |  Gray: {1/(end_cap-start_cap + end_gray-start_gray):.3f} fps")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Stoping Capture...")
close_camera(cap)
cv2.destroyAllWindows()

print("Script Done")
