# Project Overview
This repo contains code for a robotic car that identifies and follows a person. The car uses a robotic car kit with an Arduino controlling the hardware and a Raspberry Pi 3 handling the image processing and main control loop.

# File Locations
## main

## hardware_model
The hardware_model branch contains the code for the Raspberry Pi. It consists of the model generated form the main branch and can interface with the code from the hardware branch.

## hardware
The hardware branch contains the code for the Arduino. The test code folders are iterative versions of the final code and it handles all of the motor control, ultrasonic control, and interfaces with the Pi code from the hardware_model branch.
