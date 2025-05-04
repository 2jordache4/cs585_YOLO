import serial
import cv2
import numpy as np

###################### CarUnit ######################
class CarUnit:
    def __init__(self, dev, GPIO_enable=False, pin_ultra=None):
        self.dev = dev
        self.pin_ultra = pin_ultra
        self.will_collide = False
        self._GPIO_enable = GPIO_enable
        
        # Set up GPIO pins and interrupts
        if GPIO_enable:
            GPIO.setmode(GPIO.BCM)
            
            if pin_ultra is not None:
                GPIO.setup(pin_ultra, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.add_event_detect(pin_ultra, GPIO.BOTH, callback=self._interrupt_callback, bouncetime=50)
        else:
            self.pin_ultra=None
    
    def _interrupt_callback(self,channel):
        match channel:
            case self.pin_ultra:
                self.will_collide = GPIO.input(self.pin_ultra)
        
    def connect(self, baud=9600, timeout=1):
        # open dev/file
        try:
            self.usb = serial.Serial(self.dev, baud, timeout=timeout)
        except:
            print("Failed to open serial connection, script stopping")
            quit()           
        
        self.usb.reset_input_buffer()
        
    def release(self):
        # close dev/file
        # not availible for pyserial
        
        # clean up
        if self._GPIO_enable:
            self.disable_interrupts() 
            GPIO.cleanup()
        
    # disabling interupts is non reversible, and requires the object to be re-initiated
    def disable_interrupts(self): 
        if self._GPIO_enable:
            if self.pin_ultra is not None:
                GPIO.remove_event_detect(pin_ultra)
            
    def write_motors(self, r_motor=0, l_motor=0):
        # encode motor values
        signs = 0
        
        if r_motor < 0:
            signs += 1
            r_motor = -r_motor
            
        if l_motor < 0:
            signs += 2
            l_motor = -l_motor
        
        # send over serial
        self.usb.write(signs.to_bytes()+r_motor.to_bytes()+l_motor.to_bytes()+b"\n")
        
    def set_ultra(self, thresh):
        # build config msg
        #   upper nibble indicates which setting is changed,
        #   lower nibble is ignored
        config = (1 << 4) + 0
        
        # sent over serial
        self.usb.write(config.to_bytes()+thresh.to_bytes()+b"\n")

###################### SiftDetect ######################
class SIFTdetect:    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        self.target = None
        
    def match_target(self, frame, bboxes, top_count=3):
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
            return best_idx
            
            # save the first descriptors 
            # I kinda want to make this like adapting so it can save features specific to a person
            # rather than person + background but right now i think this is okay
        
        else:
            matches = self.matcher.knnMatch(self.target, des, k=2)
            
            good = [] #[m.distance < 0.8 * n.distance for m, n in matches]
            for m, n in matches: 
                if m.distance < 0.8 * n.distance: 
                    good.append([m])
                  
            if len(good) > 20:# kp threshhold, 10 was pretty consistent but im not convinced its enough
                self.target = des
                return best_idx
    
    # returns whether  a target has been saved
    def lacking_target(self):
        return self.target is None
        
    def clear_target(self):
        self.target = None
        
###################### TargetUnit ######################
class TargetUnit:
    def __init__(self, reset_func=lambda:None):
        self.target = None
        self.missing_count = 0
        
        self.reset_func = reset_func
        
    def get_motor_vals(self, new_target):
        
        if new_target is None: # no target given
            if self.target is None: # starting/reset state
                return 0,0
                
            else: # target lost state
                if self.missing_count < 4: # search phase
                    # iterate missing count
                    self.missing_count += 1
                    
                    # either keep going same way (current)
                    # or point turn in the prev target direction (add code)
                    
                elif self.missing_count < 7: # missing phase
                    self.missing_count += 1
                
                    # freeze motors
                    return 0,0
                    
                else: # reset phase
                    self.target = None
                    self.missing_count = 0
                    self.reset_func()
                    
                    return 0,0
        
        else: # target given
            # update target
            self.target = new_target
        
        
        # find motor values
        return TargetUnit._turn_val(self.target)
                
    def _turn_val(target): # straight
        DEAD_ZONE = 16
        EDGE_PWR = 70
        
        if target == 0:
            return 70,89
        
        elif target > 0: # right turn
            if target < DEAD_ZONE:
                return 70,89
        		
            l_val = 89 + int(((target-DEAD_ZONE)/160) * EDGE_PWR)
            return 70, l_val
            
        else: # left turn
            if -target < DEAD_ZONE:
                return 70,89
        		
            r_val = 70 + int(((-target-DEAD_ZONE)/160) * EDGE_PWR)
            return r_val, 89
            
            
            
            
            