import serial
import time

#motor_vals = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,240,225,210,195,180,165,150,135,120,105,90,75,60,45,30,15]
motor_vals = [0,70,90,120,150,180,210,232,255,232,210,180,150,120,90,70]

def convert_to_bytes(right_motor, left_motor):
    signs = 0
        
    if right_motor < 0:
        signs += 1
        right_motor = -right_motor
        
    if left_motor < 0:
        signs += 2
        left_motor = -left_motor
        
    return signs.to_bytes()+right_motor.to_bytes()+left_motor.to_bytes()+b"\n"

if __name__ == '__main__':
    usb = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    usb.reset_input_buffer()
    count = 0

    while True:
        for val in motor_vals:
            byte_str = convert_to_bytes(val, -val)
            
            usb.write(byte_str)
            
            print(f"{val: 4d}|{-val: 4d}|[{count}]")
            count = count+1
            time.sleep(1)
