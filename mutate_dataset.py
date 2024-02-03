from math import cos, sin
import os
from PIL import Image


INPUT_IMAGE_DIR = "/home/czack913/Code/datasets/yolo/images/val"
INPUT_ANNOTATION_DIR = "/home/czack913/Code/datasets/yolo/labels/val"
OUTPUT_IMAGE_DIR = "/home/czack913/Code/datasets/resized/temp/images"
OUTPUT_ANNOTATION_DIR = "/home/czack913/Code/datasets/resized/temp/labels"

os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(INPUT_ANNOTATION_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANNOTATION_DIR, exist_ok=True)

ANGLE_TO_RADIANS_FACTOR = 3.141592653589793 / 180.0
RADIANS = {
    45: 45 * ANGLE_TO_RADIANS_FACTOR,
    90: 90 * ANGLE_TO_RADIANS_FACTOR,
    135: 135 * ANGLE_TO_RADIANS_FACTOR,
    180: 180 * ANGLE_TO_RADIANS_FACTOR,
    225: 225 * ANGLE_TO_RADIANS_FACTOR,
    270: 270 * ANGLE_TO_RADIANS_FACTOR,
}

TRIG = {
    45: {"_sin": sin(RADIANS[45]), "_cos": cos(RADIANS[45])},
    90: {"_sin": sin(RADIANS[90]), "_cos": cos(RADIANS[90])},
    135: {"_sin": sin(RADIANS[135]), "_cos": cos(RADIANS[135])},
    180: {"_sin": sin(RADIANS[180]), "_cos": cos(RADIANS[180])},
    225: {"_sin": sin(RADIANS[225]), "_cos": cos(RADIANS[225])},
    270: {"_sin": sin(RADIANS[270]), "_cos": cos(RADIANS[270])},
}


def getAnnotationPath(jpeg_name: str):
    txt_name = os.path.splitext(jpeg_name)[0]()
    txt_filepath = os.path.join(INPUT_ANNOTATION_DIR, f"{txt_name}.txt")
    return txt_filepath


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

            x_center = x_center * TRIG[angle]._cos - y_center * TRIG[angle]._sin
            y_center = x_center * TRIG[angle]._sin + y_center * TRIG[angle]._cos

            # Write the scaled annotation to the output file
            output_file.write(
                f"{int(class_label)} {x_center} {y_center} {width} {height}\n"
            )


def transformAnnotations():
    for filename in os.listdir(INPUT_ANNOTATION_DIR):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(INPUT_ANNOTATION_DIR, filename)
            path_r45 = os.path.join(OUTPUT_ANNOTATION_DIR, f"{filename}_r45.txt")
            path_r90 = os.path.join(OUTPUT_ANNOTATION_DIR, f"{filename}_r90.txt")
            path_r135 = os.path.join(OUTPUT_ANNOTATION_DIR, f"{filename}_r135.txt")
            path_r180 = os.path.join(OUTPUT_ANNOTATION_DIR, f"{filename}_r180.txt")
            path_r225 = os.path.join(OUTPUT_ANNOTATION_DIR, f"{filename}_r225.txt")
            path_r270 = os.path.join(OUTPUT_ANNOTATION_DIR, f"{filename}_r270.txt")

            with open(input_path, "r") as input_file:
                lines = input_file.readlines()
                transformAnnotation(path_r45, lines, 45)
                transformAnnotation(path_r90, lines, 90)
                transformAnnotation(path_r135, lines, 135)
                transformAnnotation(path_r180, lines, 180)
                transformAnnotation(path_r225, lines, 225)
                transformAnnotation(path_r270, lines, 270)


def transformImages():
    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg")):
            if os.path.exists(getAnnotationPath(filename)):
                input_path = os.path.join(INPUT_IMAGE_DIR, filename)
                path_r45 = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_r45.jpeg")
                path_r90 = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_r90.jpeg")
                path_r135 = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_r135.jpeg")
                path_r180 = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_r180.jpeg")
                path_r225 = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_r225.jpeg")
                path_r270 = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_r270.jpeg")

                try:
                    with Image.open(input_path) as img:
                        img.rotate(45).save(path_r45)
                        img.rotate(90).save(path_r90)
                        img.rotate(135).save(path_r135)
                        img.rotate(180).save(path_r180)
                        img.rotate(225).save(path_r225)
                        img.rotate(270).save(path_r270)

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")


transformImages()
transformAnnotations()
