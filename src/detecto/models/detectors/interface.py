from abc import ABCMeta, abstractmethod

from pandas import DataFrame


class Detecto(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        """
        Train the anomaly detection model using the provided data.

        Parameters:
        * kwargs (DataFrame | list | str | int | float | None): The parameters depend on the detection method.

        Returns:
        * None: The result is assigned to the class attribute.
        """

    @abstractmethod
    def detect(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        """
        Predict if the provided data points are anomalies based on the trained model.

        Parameters:
        * kwargs (DataFrame | list | str | int | float | None): The parameters depend on the detection method.

        Returns:
        * None: The result is assigned to the class attribute.
        """

    @abstractmethod
    def evaluate(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        """
        Evaluate the performance of the anomaly detection model based on true and predicted labels.

        Parameters:
        * kwargs (DataFrame | list | str | int | float | None): The parameters depend on the detection method.

        Returns:
        * None: The result is assigned to the class attribute.
        """

    @property
    @abstractmethod
    def params(self) -> dict:
        """
        Retrieve the parameters of the anomaly detection model.

        Returns:
        * (dict[str | int, str | int | float | None | dict[str | int, str | int | float | None]]): Current parameters of the model.
        """

    @abstractmethod
    def set_params(self, **kwargs: str | int | float | None) -> None:
        """
        Set the parameters for the anomaly detection model.

        Parameters:
        * **kwargs (dict[str, str | int | float | None]): Parameters to be recorded from the model fitting.

        Returns:
        * None: The result is assigned to the class private attribute `__params`.
        """
