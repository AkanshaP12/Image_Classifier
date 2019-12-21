import numpy as np
import os
import cv2
from sklearn.utils import shuffle

category_names_list = ['buildings', 'forest', 'glacier', 'mountain', 'street', 'sea' ]
category_label_dict = {'buildings': 0,
                    'forest' : 1,
                    'glacier' : 2,
                    'mountain' : 3,
                    'street' : 4,
                    'sea' : 5
                    }

def load_datasets():
    datasets_types = ['train_data', 'test_data']
    datasets = []
    for dataset in datasets_types:
        datasets.append(loadImagesForDatasetName(dataset))
    print('Image datasets are loaded successfully. ' + str(len(datasets[0][0])) + ' number of train samples and ' + str(len(datasets[1][0])) + ' number of test samples.')
    return datasets

def loadImagesForDatasetName(datasetName):
    size = (150,150)
    directory = "../data/" + datasetName
    image_instances = []
    image_labels = []
    # print(directory)
    for folder in os.listdir(directory):
        # print(folder)
        current_label = category_label_dict[folder]
        # print(current_label)
        for file in os.listdir(directory + "/" + folder):
            img_path = directory + "/" + folder + "/" + file
            current_img = cv2.imread(img_path)
            # print(current_img[2][7])
            current_img = cv2.resize(current_img, size)
            image_instances.append(current_img)
            image_labels.append(current_label)
    image_instances, image_labels = shuffle(image_instances, image_labels)     ### Shuffle the data !!!
    image_instances = np.array(image_instances, dtype = 'float32') ### Our images
    image_labels = np.array(image_labels, dtype = 'int32')   ### From 0 to num_classes-1!
    
    return (image_instances, image_labels)
