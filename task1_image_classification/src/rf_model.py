from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model with any provided parameters.
        The kwargs allow to pass parameters like n_estimators, max_depth, etc.
        """
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X_train, y_train):
        """
        Train the Random Forest model using the provided training data.
        
        The MNIST images come as 28x28 arrays. Random Forest requires 1D feature vectors,
        so we reshape each image into a flat vector of 784 elements.
        
        :param X_train: Training input data of shape (n_samples, 28, 28)
        :param y_train: Training labels of shape (n_samples,)
        """
        # Flatten the images: reshape from (n_samples, 28, 28) to (n_samples, 784)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_flat, y_train)

    def predict(self, X_test):
        """
        Predict labels for the test data using the trained Random Forest model.
        
        The test images are also reshaped to match the model's expected input shape.
        
        :param X_test: Test input data of shape (n_samples, 28, 28)
        :return: Predicted labels as a numpy array
        """
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_flat)
