import cv2
import torch
import numpy as np
from collections import deque

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize tracker
class SimpleTracker:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        for tracker in self.trackers:
            tracker['lost'] += 1

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            matched = False
            for tracker in self.trackers:
                if self.iou(tracker['bbox'], [x1, y1, x2, y2]) > 0.5:
                    tracker['bbox'] = [x1, y1, x2, y2]
                    tracker['lost'] = 0
                    matched = True
                    break
            if not matched:
                self.trackers.append({'bbox': [x1, y1, x2, y2], 'lost': 0})

        self.trackers = [t for t in self.trackers if t['lost'] < 5]

        return self.trackers

    def iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_ - x1_) * (y2_ - y1_)

        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area

# Calculate Euclidean distance between two vehicles
def calculate_distance(vehicle1, vehicle2):
    pos1 = np.array(vehicle1['center'])
    pos2 = np.array(vehicle2['center'])
    distance = np.linalg.norm(pos1 - pos2)
    return distance

# Process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = SimpleTracker()
    vehicle_tracks = {}
    warning_distance = 50  # Threshold distance in pixels

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_with_warnings.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if cls == 2:  # Only consider cars
                detections.append(box + [conf, cls])

        tracks = tracker.update(detections)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2 = map(int, track['bbox'])
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            track['center'] = center
            if i not in vehicle_tracks:
                vehicle_tracks[i] = deque(maxlen=5)
            vehicle_tracks[i].append(center)
            if len(vehicle_tracks[i]) > 1:
                track['velocity'] = np.mean(np.diff(vehicle_tracks[i], axis=0), axis=0)
            else:
                track['velocity'] = np.array([0, 0])

            # Draw bounding box and center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(center[0]), int(center[1])), 2, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {i}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i, track in enumerate(tracks):
            for j, other_track in enumerate(tracks):
                if i != j:
                    distance = calculate_distance(track, other_track)
                    if distance < warning_distance:
                        cv2.putText(frame, f'Warning! Vehicles {i} and {j} too close!', (50, 50 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = 'C:Users\tejas\downloads\vehicle_detection\video.mp4'  # Replace with your local video file path
    process_video(video_path)
