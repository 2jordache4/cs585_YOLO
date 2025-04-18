import serial
import time

if __name__ == '__main__':
	ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
	ser.reset_input_buffer()
	
	val = 0
	sign = -1
	
	val1 = 0
	val2 = 0

	while True:
		signs = 0
		
		if val1 < 0:
			signs += 1
			val1 = -val1
			
		if val2 < 0:
			signs += 2
			val2 = -val2
		ser.write(signs.to_bytes()+val1.to_bytes()+val2.to_bytes()+b"\n")
		line = ser.readline().decode('utf-8').rstrip()
		print(line)
		time.sleep(1)
		
		val = (val + 5) % 256
		sign = -sign
		
		val1 = val * sign
		val2 = val * -sign
