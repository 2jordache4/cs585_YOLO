import os
import json
import shutil
from tqdm import tqdm

"""
This file took the COCO dataset, and pruned out images and labels we did not need to create a custom 
dataset of images either containing humans, or not.
"""

def prepare_yolo_dataset(
    annotation_path,
    image_dir,
    output_image_dir,
    output_label_dir,
    category_id=1,
    empty_ratio=0.15):
    # COCO annotations
    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # image lookup
    image_lookup = {
        img["id"]: (img["file_name"], img["width"], img["height"])
        for img in coco["images"]
    }

    # group person annotations by image_id
    annotations_by_image = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != category_id:
            continue
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    # copy images with people and create labels for 'new' dataset
    for img_id, anns in tqdm(annotations_by_image.items()):
        file_name, img_w, img_h = image_lookup[img_id]
        src_img_path = os.path.join(image_dir, file_name)
        dst_img_path = os.path.join(output_image_dir, file_name)
        dst_label_path = os.path.join(output_label_dir, os.path.splitext(file_name)[0] + ".txt")

        if not os.path.exists(src_img_path):
            continue

        shutil.copy(src_img_path, dst_img_path)

        with open(dst_label_path, "w") as f:
            for ann in anns:
                x, y, w, h = ann["bbox"]
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    # process empty (no-person) images 
    person_image_ids = set(annotations_by_image.keys())
    empty_images = [img for img in coco["images"] if img["id"] not in person_image_ids]
    num_empty_to_copy = int(len(annotations_by_image) * empty_ratio)

    count = 0
    for img in tqdm(empty_images):
        if count >= num_empty_to_copy:
            break
        file_name = img["file_name"]
        src_path = os.path.join(image_dir, file_name)
        dst_path = os.path.join(output_image_dir, file_name)
        label_path = os.path.join(output_label_dir, os.path.splitext(file_name)[0] + ".txt")

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            with open(label_path, 'w') as lf:
                lf.write('')
            count += 1


prepare_yolo_dataset(
    annotation_path='./coco_dataset/annotations/instances_train2017.json',
    image_dir = './coco_dataset/train2017',
    output_image_dir='./custom_dataset/images/train',
    output_label_dir='./custom_dataset/labels/train',
    category_id=1,  # person
    empty_ratio=0.15
)

prepare_yolo_dataset(
    annotation_path='./coco_dataset/annotations/instances_val2017.json',
    image_dir='./coco_dataset/val2017',
    output_image_dir='./custom_dataset/images/val',
    output_label_dir='./custom_dataset/labels/val',
    category_id=1,  
    empty_ratio=0.15
)

