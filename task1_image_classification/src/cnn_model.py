import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from interface import MnistClassifierInterface

class CNNModel(MnistClassifierInterface):
    """
    Convolutional Neural Network (CNN) for MNIST classification.
    """
    def __init__(self, num_classes=10, epochs=10, batch_size=32):
        """
        Initialize the CNN model.
        - Uses Convolutional Layers for feature extraction.
        - MaxPooling for dimensionality reduction.
        - Dropout & BatchNormalization for improved training stability.
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        # Define CNN architecture
        self.model = Sequential([
            Input(shape=(28, 28, 1)),  
            Conv2D(32, kernel_size=(3,3), activation='relu'),
            BatchNormalization(),  # Improves learning stability
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(64, kernel_size=(3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),

            Flatten(),  
            Dense(128, activation='relu'),
            Dropout(0.3),  # Helps prevent overfitting  
            Dense(num_classes, activation='softmax')  
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train):
        """
        Train the CNN model.
        - Converts labels to one-hot encoding.
        - Fits the model using provided training data.
        """
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X_test):
        """
        Predict class labels using the trained CNN model.
        - Returns class labels instead of softmax probabilities.
        """
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)
