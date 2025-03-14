import cv2
import shutil
import numpy as np
from pathlib import Path

def apply_color_filter(image, color):
    result = image.copy()

    if color == 'blue':
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.5, 0, 255)
    elif color == 'yellow':
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.7, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.2, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255)
    elif color == 'green':
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.5, 0, 255)
    elif color == 'red':
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.5, 0, 255)
    elif color == 'orange':
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.2, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.5, 0, 255)

    return result

def generate_augmented_images():
    images_dir = Path('val/images')
    labels_dir = Path('val/labels')

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: folders {images_dir} and {labels_dir} must exist")
        return

    for dir_path in [images_dir, labels_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

    colors = ['blue', 'yellow', 'green', 'red', 'orange']

    for img_file in images_dir.glob('*'):
        if not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        label_file = labels_dir / f"{img_file.stem}.txt"

        img = cv2.imread(str(img_file))

        for color in colors:
            filtered_img = apply_color_filter(img, color)

            new_img_name = f"{img_file.stem}_{color}{img_file.suffix}"
            new_img_path = images_dir / new_img_name
            cv2.imwrite(str(new_img_path), filtered_img)
            new_label_name = f"{img_file.stem}_{color}.txt"
            new_label_path = labels_dir / new_label_name
            shutil.copy(str(label_file), str(new_label_path))

            print(f"Created {new_img_name} with {color} filter")

if __name__ == "__main__":
    generate_augmented_images()