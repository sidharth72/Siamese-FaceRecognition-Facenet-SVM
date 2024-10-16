import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

def detect_and_crop_face(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    
    if faces:
        x, y, width, height = faces[0]['box']
        face = image[y: y + height, x: x + width]
        return cv2.resize(face, (160, 160))

    return None

def preprocess_image(img):
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img



