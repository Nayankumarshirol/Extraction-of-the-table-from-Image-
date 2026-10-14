import os

import cv2
import numpy as np


# ========== CONFIG ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = r"F:\Cogniquest\table extraction frm img\Final_images\Final_images"
OUTPUT_FOLDER = os.path.join(BASE_DIR, "separated_tables_filtered")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ========== HELPER FUNCTIONS ==========
def is_skewed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return False

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    avg_angle = np.mean(angles)
    return abs(avg_angle) > 5


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh


def detect_table_structure(thresh):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    table_mask = cv2.add(horizontal, vertical)
    return table_mask


def extract_table_regions(image, mask):
    image_height, image_width = image.shape[:2]
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, image_width // 25), max(30, image_height // 25)),
    )
    merged_mask = cv2.dilate(mask, merge_kernel, iterations=2)

    contours, _ = cv2.findContours(
        merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidate_tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_ratio = area / float(image_width * image_height)
        width_ratio = w / float(image_width)
        height_ratio = h / float(image_height)

        # Keep only substantial table blocks and reject narrow side fragments.
        if (
            area_ratio >= 0.08
            and width_ratio >= 0.35
            and height_ratio >= 0.18
            and max(width_ratio, height_ratio) >= 0.45
        ):
            pad = 12
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, image.shape[1])
            y2 = min(y + h + pad, image.shape[0])
            table_crop = image[y1:y2, x1:x2]
            candidate_tables.append((x1, y1, x2 - x1, y2 - y1, table_crop))

    candidate_tables.sort(key=lambda b: (b[2] * b[3]), reverse=True)

    filtered_tables = []
    for candidate in candidate_tables:
        x, y, w, h, crop = candidate
        is_inside_larger_table = False

        for existing_x, existing_y, existing_w, existing_h, _ in filtered_tables:
            inside_horizontally = x >= existing_x and (x + w) <= (existing_x + existing_w)
            inside_vertically = y >= existing_y and (y + h) <= (existing_y + existing_h)
            if inside_horizontally and inside_vertically:
                is_inside_larger_table = True
                break

        if not is_inside_larger_table:
            filtered_tables.append((x, y, w, h, crop))

    return filtered_tables


def image_hash(image):
    if image is None or image.size == 0:
        return None

    resized = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray.tobytes()


def save_table_crop(table_crop, filename):
    if table_crop is None or table_crop.size == 0:
        return
    cv2.imwrite(filename, table_crop)


def iter_image_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for img_name in sorted(files):
            img_path = os.path.join(root, img_name)
            if os.path.splitext(img_name)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            yield img_name, img_path


def build_output_path(img_path):
    relative_path = os.path.relpath(img_path, INPUT_FOLDER)
    relative_without_ext = os.path.splitext(relative_path)[0]
    safe_name = relative_without_ext.replace(os.sep, "__").replace(" ", "_")
    return os.path.join(OUTPUT_FOLDER, f"{safe_name}")


# ========== MAIN PIPELINE ==========
def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    processed_hashes = set()

    for img_name, img_path in iter_image_files(INPUT_FOLDER):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable file: {img_name}")
            continue

        thresh = preprocess(image)
        mask = detect_table_structure(thresh)
        tables = extract_table_regions(image, mask)

        if not tables:
            print(f"No table regions detected: {img_name}")
            continue

        base_output = build_output_path(img_path)
        saved_any = False

        for index, (_, _, _, _, table_crop) in enumerate(tables, start=1):
            table_hash = image_hash(table_crop)
            if table_hash is None:
                continue
            if table_hash in processed_hashes:
                print(f"Skipping duplicate table: {img_name} [{index}]")
                continue
            processed_hashes.add(table_hash)

            output_file = f"{base_output}_table_{index}.png"
            if os.path.exists(output_file):
                print(f"Skipping existing output: {output_file}")
                continue

            save_table_crop(table_crop, output_file)
            saved_any = True
            print(f"Saved: {output_file}")

        if not saved_any:
            print(f"No new table images saved for: {img_name}")

    print(f"All tables processed. Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
