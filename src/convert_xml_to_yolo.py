"""
Convert NEU-DET Pascal VOC XML annotations to YOLO format.

NEU-DET structure:
  NEU-DET/
    train/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/
    train/annotations/*.xml
    validation/images/...
    validation/annotations/...

Output: data/{train,valid}/images/ + data/{train,valid}/labels/
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SRC = Path("/tmp/cv_project/NEU-DET")
DST = Path("/home/arihant/neu-defect-detection/data")

# Class mapping (NEU-DET steel surface defects)
CLASSES = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def parse_xml(xml_path):
    """Parse Pascal VOC XML, return list of (class_id, x, y, w, h) in normalized coords."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue
        class_id = CLASSES[name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Normalize to [0, 1]
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes, (img_w, img_h)


def convert_split(split_name):
    """Convert one split (train/validation) to YOLO format."""
    src_images = SRC / split_name / "images"
    src_annots = SRC / split_name / "annotations"

    dst_images = DST / split_name / "images"
    dst_labels = DST / split_name / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Find all images (they're in class subdirectories)
    image_files = []
    for cls_dir in sorted(src_images.iterdir()):
        if cls_dir.is_dir():
            for img_path in cls_dir.glob("*.jpg"):
                image_files.append(img_path)

    print(f"[{split_name}] Found {len(image_files)} images")

    converted = 0
    missing_xml = 0
    for img_path in image_files:
        # XML has same basename but .xml extension
        xml_path = src_annots / (img_path.stem + ".xml")
        if not xml_path.exists():
            missing_xml += 1
            continue

        boxes, (img_w, img_h) = parse_xml(xml_path)
        if not boxes:
            continue

        # Copy image
        dst_img = dst_images / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write YOLO label
        dst_label = dst_labels / (img_path.stem + ".txt")
        with open(dst_label, "w") as f:
            for class_id, xc, yc, w, h in boxes:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        converted += 1

    print(f"[{split_name}] Converted {converted} images ({missing_xml} missing XML)")
    return converted


if __name__ == "__main__":
    print("=" * 60)
    print("NEU-DET: Steel Surface Defect Detection")
    print("Converting Pascal VOC XML → YOLO format")
    print("=" * 60)

    total = 0
    total += convert_split("train")
    total += convert_split("validation")

    print(f"\nTotal: {total} images converted")
    print(f"Output: {DST}")

    # Print class distribution
    print("\nClass distribution:")
    for cls_name, cls_id in CLASSES.items():
        count = 0
        for split in ["train", "valid"]:
            label_dir = DST / split / "labels"
            for label_file in label_dir.glob("*.txt"):
                with open(label_file) as f:
                    count += sum(1 for line in f if line.strip().startswith(f"{cls_id} "))
        print(f"  {cls_name}: {count} instances")