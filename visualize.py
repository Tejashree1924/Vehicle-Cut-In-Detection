import cv2
import matplotlib.pyplot as plt
from detect_objects import detect_objects
from estimate_depth import estimate_depth
from track_objects import track_objects

def calculate_distance(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    distance = depth_map[int(center_y), int(center_x)]
    return distance

def calculate_ttc(distance, velocity):
    if velocity == 0:
        return float('inf')
    ttc = distance / velocity
    return ttc

def visualize(image, detections, depth_map, threshold=0.7):
    for detection in detections:
        bbox, confidence, class_id = detection[:4], detection[4], int(detection[5])
        if confidence < 0.5:
            continue
        distance = calculate_distance(depth_map, bbox)
        ttc = calculate_ttc(distance, velocity=1)
        if ttc < threshold:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(image, 'WARNING', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, f'TTC: {ttc:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    image_path = '../data/processed_data/sample.jpg'
    image = cv2.imread(image_path)
    detections = detect_objects(image).xywh[0].cpu().numpy()
    depth_map = estimate_depth(Image.open(image_path).convert('RGB')).squeeze().numpy()
    tracks = track_objects(detections)
    visualize(image, tracks, depth_map)
