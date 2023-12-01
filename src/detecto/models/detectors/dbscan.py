from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto


class DBSCANDetecto(Detecto):
    """
    Anomaly detector class that implements the "Density-Based Spatial Clustering of Applications with Noise" (D. B. S. C. A. N.) method.

    # Attributes
    ------------
    ! TODO: Implement anomaly detection with "DBSCAN" method!
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
        return "Density-Based Spatial Clustering of Applications with Noise Anomaly Detector"
