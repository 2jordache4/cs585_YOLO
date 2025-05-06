import os
import cv2
"""
This file was used to convert the custom dataset from RGB to grayscale.
"""

src_path = './custom_dataset/images/val'
dst_path = './greyscale_dataset/images/val'

os.makedirs(dst_path, exist_ok=True)
files = os.listdir(src_path)

for file_name in files:
    src_file = os.path.join(src_path, file_name)
    dst_file = os.path.join(dst_path, file_name)

    if os.path.isfile(src_file) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img = cv2.imread(src_file)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(dst_file, gray)
        else:
            print(f"Warning: Couldn't read image {src_file}")
