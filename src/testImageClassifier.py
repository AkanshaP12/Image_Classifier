import numpy as np
import matplotlib.pyplot as plt
from imageClassifier import ImageClassifier
import loadImageData as imageDataset
import loadPredictSamples as predictDataset

def load_scaled_data():
    #load train and test datasets
    (images_train, train_labels), (images_test, test_labels) = imageDataset.load_datasets()

    # scaling of data
    images_train = images_train / 255.0 
    images_test = images_test / 255.0
    return images_train, train_labels, images_test, test_labels

def predictOnNewImage(imgClassifier):
    X_pred = predictDataset.loadPredictSamples(count=12)
    # scaling of data
    X_pred = X_pred / 255.0
    y_class_indices = imgClassifier.predict(X_pred)
    y_pred_labels = [imageDataset.category_names_list[image_category_prediction] for image_category_prediction in y_class_indices]

    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Predicted labels for the images:", fontsize=16)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_pred[i], cmap=plt.cm.binary)
        plt.xlabel(y_pred_labels[i])
    plt.show()


X_train, y_train, X_test, y_test = load_scaled_data()

imgClassifier = ImageClassifier(metrics=['accuracy'])
imgClassifier.fit(X_train, y_train, batch_size=128, epochs=4, verbose=2)

imgClassifier.evaluateAndScore(X_test, y_test)
predictOnNewImage(imgClassifier)

