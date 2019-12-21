import numpy as np
import tensorflow as tf

class ImageClassifier:
    def __init__(self, input_shape=(150, 150, 3), optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=[]):
        # Building model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape), # the nn will learn the good filter to use
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(6, activation=tf.nn.softmax)
        ])
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, X, y, batch_size=128, epochs=10, validation_split= 0.2, verbose=1):
        self.model.compile(optimizer= self.optimizer, loss= self.loss, metrics=self.metrics)
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=verbose)

    def predict(self, X):
        predicted_probabilities = self.model.predict(X)
        # take highest probabilitzy to determine class of instance
        y_predicted = [np.argmax(image_probabilities) for image_probabilities in predicted_probabilities]
        return y_predicted

    def evaluateAndScore(self, X, y):
        return self.model.evaluate(X, y)