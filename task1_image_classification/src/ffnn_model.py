import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from interface import MnistClassifierInterface

class FFNNModel(MnistClassifierInterface):
    def __init__(self, input_shape=(28,28), num_classes=10, epochs=10, batch_size=32):
        """
        Initialize the Feed-Forward Neural Network (FFNN).
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        # Define the neural network architecture
        self.model = Sequential([
            Flatten(input_shape=input_shape),  # Convert 28x28 image into a 1D vector
            Dense(128, activation='relu'),  # First hidden layer
            Dense(num_classes, activation='softmax')  # Output layer (10 classes)
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train):
        """
        Train the FFNN model.
        """
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)  # Convert labels to one-hot encoding
        self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X_test):
        """
        Predict labels using the trained FFNN model.
        """
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)  # Convert softmax output to class labels


