from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto


class OneClassSVMDetecto(Detecto):
    """
    Anomaly detector class that implements the "Once Class Support Vector Machine" (S. V. M.) method.

    # Attributes
    ------------
    ! TODO: Implement anomaly detection with SVM method!
    """

    def fit(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        pass

    def detect(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        pass

    def evaluate(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        pass

    @property
    def params(self) -> dict:  # type: ignore
        pass

    def set_params(self, **kwargs: str | int | float | None) -> None:
        pass

    def __str__(self) -> str:
        return "One Class Support Vector Machine Anomaly Detector"
