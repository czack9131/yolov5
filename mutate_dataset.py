from math import cos, sin
import os
import shutil
from PIL import Image

BASE_DIR = "/home/czack913/Code"


INPUT_IMAGE_DIR = f"{BASE_DIR}/datasets/yolo/images/train"
OUTPUT_IMAGE_DIR = f"{BASE_DIR}/datasets/yolo/images/out"

INPUT_ANNOTATION_DIR = f"{BASE_DIR}/datasets/yolo/labels/train"
OUTPUT_ANNOTATION_DIR = f"{BASE_DIR}/datasets/yolo/labels/out"

MUTATED_IMAGE_DIR = f"{BASE_DIR}/datasets/resized/temp/images"
MUTATED_ANNOTATION_DIR = f"{BASE_DIR}/datasets/resized/temp/labels"

# os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
# os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
# os.makedirs(MUTATED_IMAGE_DIR, exist_ok=True)
# os.makedirs(INPUT_ANNOTATION_DIR, exist_ok=True)
# os.makedirs(OUTPUT_ANNOTATION_DIR, exist_ok=True)
# os.makedirs(MUTATED_ANNOTATION_DIR, exist_ok=True)

ANGLE_TO_RADIANS_FACTOR = 3.141592653589793 / 180.0
RADIANS = {
    45: 45 * ANGLE_TO_RADIANS_FACTOR,
    90: 90 * ANGLE_TO_RADIANS_FACTOR,
    135: 135 * ANGLE_TO_RADIANS_FACTOR,
    180: 180 * ANGLE_TO_RADIANS_FACTOR,
    225: 225 * ANGLE_TO_RADIANS_FACTOR,
    270: 270 * ANGLE_TO_RADIANS_FACTOR,
}


class Annotation:
    label: int
    cx: float
    cy: float
    w: float
    h: float

    def rotate90(self):
        return (self.label, self.cy, 1.0 - self.cx, self.h, self.w)

    def rotate180(self):
        return (self.label, 1.0 - self.cx, 1.0 - self.cy, self.w, self.h)

    def rotate270(self):
        return (self.label, 1.0 - self.cy, self.cx, self.h, self.w)


def reformatNames(in_dir: str, out_dir: str, exts: tuple[str]):
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(in_dir):
        input_path = os.path.join(in_dir, filename)
        lower = filename.lower()
        if lower.endswith(exts):
            ts_part = filename.split("_")[1]
            part = ts_part[:14]

            (_, ext) = os.path.splitext(lower)
            if ext == ".jpg":
                ext = ".jpeg"

            output_fn = f"images_{part}{ext}"
            output_path = os.path.join(out_dir, output_fn)

            shutil.copy(input_path, output_path)


def getAnnotationPath(jpeg_name: str, output_dir: str) -> str:
    txt_name = os.path.splitext(jpeg_name)[0]
    txt_filepath = os.path.join(output_dir, f"{txt_name}.txt")
    return txt_filepath


def transformAnnotations(input_dir: str, output_dir: str):
    for filename in os.listdir(input_dir):
        lower = filename.lower()
        (base, ext) = os.path.splitext(lower)
        if ext == ".txt":
            input_path = os.path.join(input_dir, lower)

            for angle in [90, 180, 270]:
                output_path = os.path.join(output_dir, f"{base}_r{angle}{ext}")

                with open(input_path, "r") as input_file:
                    lines = input_file.readlines()
                    transformAnnotation(output_path, lines, angle)


def transformImages(input_dir: str, output_dir: str):
    for filename in os.listdir(input_dir):
        lower = filename.lower()
        (base, ext) = os.path.splitext(lower)

        if ext in (".jpg", ".jpeg"):
            input_path = os.path.join(input_dir, lower)

            for angle in [90, 180, 270]:
                output_path = os.path.join(output_dir, f"{base}_r{angle}.jpeg")

                try:
                    with Image.open(input_path) as img:
                        img.rotate(angle).save(output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")


def transformAnnotation(output_path: str, lines: list[str], angle: float):
    with open(output_path, "w") as output_file:
        for line in lines:
            if not line.strip():
                continue

            values = line.split()

            # Check if the line has the expected number of values (5)
            if len(values) != 5:
                print(f"Skipping line: {line}")
                continue

            # Extract values
            class_label, x_center, y_center, width, height = map(float, values)
            if angle == 90:
                new_x_center = y_center
                new_y_center = 1.0 - x_center
                new_width = height
                new_height = width
            elif angle == 180:
                new_x_center = 1.0 - x_center
                new_y_center = 1.0 - y_center
                new_width = width
                new_height = height
            elif angle == 270:
                new_x_center = 1.0 - y_center
                new_y_center = x_center
                new_width = height
                new_height = width

            # Write the scaled annotation to the output file
            output_file.write(
                f"{int(class_label)} {new_x_center} {new_y_center} {new_width} {new_height}\n"
            )


# reformatNames(INPUT_ANNOTATION_DIR, OUTPUT_ANNOTATION_DIR, (".txt", ".jpg", ".jpeg"))
# reformatNames(INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR, (".txt", ".jpg", ".jpeg"))
# reformatNames(INPUT_ANNOTATION_DIR, OUTPUT_ANNOTATION_DIR, (".txt", ".jpg", ".jpeg"))

A = f"{BASE_DIR}/datasets/yolo/images/train"
B = f"{BASE_DIR}/datasets/yolo/images/val"
C = f"{BASE_DIR}/datasets/yolo/images/test"
AA = f"{BASE_DIR}/datasets/resized/images/train"
BB = f"{BASE_DIR}/datasets/resized/images/val"
CC = f"{BASE_DIR}/datasets/resized/images/test"

D = f"{BASE_DIR}/datasets/yolo/labels/train"
E = f"{BASE_DIR}/datasets/yolo/labels/val"
F = f"{BASE_DIR}/datasets/yolo/labels/test"
DD = f"{BASE_DIR}/datasets/resized/labels/train"
EE = f"{BASE_DIR}/datasets/resized/labels/val"
FF = f"{BASE_DIR}/datasets/resized/labels/test"

os.makedirs(A, exist_ok=True)
os.makedirs(AA, exist_ok=True)
os.makedirs(B, exist_ok=True)
os.makedirs(BB, exist_ok=True)
os.makedirs(C, exist_ok=True)
os.makedirs(CC, exist_ok=True)
os.makedirs(D, exist_ok=True)
os.makedirs(DD, exist_ok=True)
os.makedirs(E, exist_ok=True)
os.makedirs(EE, exist_ok=True)
os.makedirs(F, exist_ok=True)
os.makedirs(FF, exist_ok=True)

transformImages(A, AA)
transformImages(B, BB)
transformImages(C, CC)

transformAnnotations(D, DD)
transformAnnotations(E, EE)
transformAnnotations(F, FF)
