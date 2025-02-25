from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the classifier using the provided training data.
        
        :param X_train: The training input data.
        :param y_train: The training labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict labels for the given test data.
        
        :param X_test: The test input data.
        :return: The predicted labels.
        """
        pass
