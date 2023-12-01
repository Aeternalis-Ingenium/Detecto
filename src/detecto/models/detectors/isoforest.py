from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto


class IsoForestDetecto(Detecto):
    """
    Anomaly detector class that implements the "Isolation Forest" method.

    # Attributes
    ------------
    ! TODO: Implement anomaly detection with isolation forest method!
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
        return "Isolation Forest Anomaly Detector"
