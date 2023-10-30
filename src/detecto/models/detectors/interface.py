from abc import ABCMeta, abstractmethod

from pandas import DataFrame


class Detecto(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, dataset: DataFrame | list) -> None:
        """
        Train the anomaly detection model using the provided data.

        Parameters:
        - dataset (DataFrame or array-like): Input data for training.

        Returns:
        - None
        """

    @abstractmethod
    def score(self, dataset: DataFrame | list) -> DataFrame:
        """
        Compute the anomaly scores for the provided data based on the trained model.

        Parameters:
        - dataset (DataFrame or array-like): Data for which anomaly scores are to be computed.

        Returns:
        - scores (DataFrame): Anomaly scores for each data point.
        """

    @abstractmethod
    def detect(self, dataset: DataFrame | list) -> DataFrame:
        """
        Predict if the provided data points are anomalies based on the trained model.

        Parameters:
        - dataset (DataFrame or array-like): Data for which predictions are to be made.

        Returns:
        - predictions (DataFrame): Binary labels indicating anomalies (1 for anomaly, 0 otherwise).
        """

    @abstractmethod
    def evaluate(self, dataset: DataFrame | list, detected: DataFrame | list) -> DataFrame:
        """
        Evaluate the performance of the anomaly detection model based on true and predicted labels.

        Parameters:
        - dataset (DataFrame or array-like): Data for which predictions are to be made.
        - detected (DataFrame | array-like): Detected labels from the model.

        Returns:
        - metrics (DataFrame): Performance metrics.
        """

    @abstractmethod
    def set_params(self, **params: dict) -> None:
        """
        Set the parameters for the anomaly detection model.

        Parameters:
        - **params (dict): Parameters to be set for the model.

        Returns:
        - None
        """

    @abstractmethod
    def get_params(self) -> dict:
        """
        Retrieve the parameters of the anomaly detection model.

        Returns:
        - params (dict): Current parameters of the model.
        """
