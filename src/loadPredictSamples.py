import numpy as np
import os
import cv2
from sklearn.utils import shuffle

def loadPredictSamples(count=10):
    size = (150,150)
    directory = "../data/pred_data"
    image_instances = []
    loop_counter = 0
    for file in os.listdir(directory):
        if loop_counter == count:
            break
        loop_counter += 1
        img_path = directory + "/" + file
        current_img = cv2.imread(img_path)
        current_img = cv2.resize(current_img, size)
        image_instances.append(current_img)

    image_instances = shuffle(image_instances)
    return np.array(image_instances, dtype = 'float32')