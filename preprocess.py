import cv2
import os
import numpy as np

def preprocess_image(image, target_size=(640, 480)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_images(image_dir, target_size=(640, 480)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            preprocessed_image = preprocess_image(image, target_size)
            images.append(preprocessed_image)
    return np.array(images)
