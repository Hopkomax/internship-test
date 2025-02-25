import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from interface import MnistClassifierInterface

class CNNModel(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, epochs=10, batch_size=32):
        """
        Convolutional Neural Network (CNN) for MNIST classification.
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        # Define CNN architecture
        self.model = Sequential([
            Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(64, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),

            Flatten(),  
            Dense(128, activation='relu'),
            Dropout(0.3),  
            Dense(num_classes, activation='softmax')  
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train):
        """
        Train the CNN model.
        """
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)  # Convert labels to one-hot encoding
        self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X_test):
        """
        Predict labels using the trained CNN model.
        """
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)  # Convert softmax output to class labels
