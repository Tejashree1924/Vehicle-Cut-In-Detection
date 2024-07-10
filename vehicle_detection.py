import cv2
import numpy as np
import math
import subprocess

def reencode_video(input_file, output_file):
    cmd = f"ffmpeg -i {input_file} -c:v copy -c:a copy -movflags faststart {output_file}"
    subprocess.call(cmd, shell=True)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = np.copy(img)
    if lines is None:
        return img
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_vehicles(image):
    results = model(image)
    detections = results.pandas().xyxy[0]
    
    vehicles = []
    for _, row in detections.iterrows():
        if row['name'] in ["car", "bus", "truck", "motorcycle"]:
            x, y, w, h = int(row['xmin']), int(row['ymin']), int(row['xmax']) - int(row['xmin']), int(row['ymax']) - int(row['ymin'])
            vehicles.append((x, y, w, h))
            label = row['name']
            color = (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vehicles
def highlight_vehicle_area(image, left_line, right_line, vehicles):
    overlay = image.copy()
    warning = False
    if left_line is not None and right_line is not None:
        height = image.shape[0]
        min_y = int(height * (3 / 5))
        max_y = int(height)

        left_x_start, left_y_start, left_x_end, left_y_end = left_line
        right_x_start, right_y_start, right_x_end, right_y_end = right_line

        pts = np.array([
            [left_x_start, left_y_start],
            [left_x_end, left_y_end],
            [right_x_end, right_y_end],
            [right_x_start, right_y_start]
        ], np.int32)

        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        for (x, y, w, h) in vehicles:
            vehicle_center = (float(x + w // 2), float(y + h // 2))
            if cv2.pointPolygonTest(pts, vehicle_center, False) >= 0:
                warning = True

    if warning:
        cv2.putText(image, "Caution!! Vehicle Cutting In!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image
def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=20, maxLineGap=300)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:  # Prevent division by zero
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 0.5:
                    continue
                if slope <= 0:
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

    left_line = None
    right_line = None
    if left_line_x and left_line_y and right_line_x and right_line_y:
        min_y = int(image.shape[0] * (3 / 5))
        max_y = int(image.shape[0])

        if left_line_x and left_line_y:
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
            left_line = (int(poly_left(max_y)), max_y, int(poly_left(min_y)), min_y)

        if right_line_x and right_line_y:
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
            right_line = (int(poly_right(max_y)), max_y, int(poly_right(min_y)), min_y)

        line_image = draw_lines(image, [[left_line, right_line]], thickness=5)
    else:
        line_image = image  # If no lines are detected, return the original image

    vehicles = detect_vehicles(line_image)

    final_image = highlight_vehicle_area(line_image, left_line, right_line, vehicles)

    return final_image
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_detection(detected_vehicles, ground_truth_vehicles, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_ground_truths = []

    for gt_vehicle in ground_truth_vehicles:
        gt_box = (gt_vehicle[0], gt_vehicle[1], gt_vehicle[0] + gt_vehicle[2], gt_vehicle[1] + gt_vehicle[3])
        matched = False
        for det_vehicle in detected_vehicles:
            det_box = (det_vehicle[0], det_vehicle[1], det_vehicle[0] + det_vehicle[2], det_vehicle[1] + det_vehicle[3])
            if calculate_iou(gt_box, det_box) >= iou_threshold:
                matched = True
                break

        if matched:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = len(detected_vehicles) - true_positives

    return true_positives, false_positives, false_negatives

def read_ground_truth_from_csv(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    if 'labels' not in df.columns:
        raise ValueError("CSV file must contain a 'labels' column.")
    if 'x_min' not in df.columns or 'y_min' not in df.columns or 'x_max' not in df.columns or 'y_max' not in df.columns:
        raise ValueError("CSV file must contain 'x_min', 'y_min', 'x_max', and 'y_max' columns.")
    
    labels = df['labels'].unique()
    ground_truth_data = {label: [] for label in labels}
    for _, row in df.iterrows():
        label = row['labels']
        x_min = int(row['x_min'])
        y_min = int(row['y_min'])
        x_max = int(row['x_max'])
        y_max = int(row['y_max'])
        box = (x_min, y_min, x_max - x_min, y_max - y_min)
        ground_truth_data[label].append(box)
    return ground_truth_data

