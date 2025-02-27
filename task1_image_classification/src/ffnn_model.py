from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from interface import MnistClassifierInterface
import numpy as np

class FFNNModel(MnistClassifierInterface):
    def __init__(self, num_classes=10, epochs=10, batch_size=32):
        """
        Initialize the Feed-Forward Neural Network (FFNN).
        """
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        # ✅ Correct FFNN Model Architecture (Flattened input)
        self.model = Sequential([
            Input(shape=(784,)),  # 🛠️ Correct input shape: Flattened 28x28 -> 784
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 🔍 Debugging: Print expected input shape
        print(f"✅ FFNN Model Built. Expected Input Shape: {self.model.input_shape}")

    def train(self, X_train, y_train):
        """
        Train the FFNN model.
        """
        # 🔍 Debugging: Print input shape before training
        print(f"🔹 Training Data Shape: {X_train.shape}")

        # Convert labels to one-hot encoding
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)

        # 🔍 Debugging: Ensure one-hot encoding is correct
        print(f"🔹 y_train one-hot encoded shape: {y_train_cat.shape}")

        # Train the model
        self.model.fit(X_train, y_train_cat, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X_test):
        """
        Predict labels using the trained FFNN model.
        """
        # 🔍 Debugging: Print input shape before prediction
        print(f"🔹 Test Data Shape: {X_test.shape}")

        # Generate predictions
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)  # Convert softmax output to class labels.