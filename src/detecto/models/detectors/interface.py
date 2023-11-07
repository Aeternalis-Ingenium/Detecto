from abc import ABCMeta, abstractmethod, abstractproperty

from pandas import DataFrame


class Detecto(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        """
        Train the anomaly detection model using the provided data.

        Parameters:
        - dataset (DataFrame or array-like): Input data for training.

        Returns:
        - None
        """

    @abstractmethod
    def detect(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        """
        Predict if the provided data points are anomalies based on the trained model.

        Parameters:
        - dataset (DataFrame or array-like): Data for which predictions are to be made.

        Returns:
        - predictions (DataFrame): Binary labels indicating anomalies (1 for anomaly, 0 otherwise).
        """

    @abstractmethod
    def evaluate(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        """
        Evaluate the performance of the anomaly detection model based on true and predicted labels.

        Parameters:
        - dataset (DataFrame or array-like): Data for which predictions are to be made.
        - detected (DataFrame | array-like): Detected labels from the model.

        Returns:
        - metrics (DataFrame): Performance metrics.
        """

    @abstractmethod
    def set_params(self, **kwargs: int | float | None) -> None:
        """
        Set the parameters for the anomaly detection model.

        Parameters:
        - **kwargs (dict): Parameters to be set for the model.

        Returns:
        - None
        """

    @abstractproperty
    def get_params(self) -> dict:
        """
        Retrieve the parameters of the anomaly detection model.

        Returns:
        - params (dict): Current parameters of the model.
        """
