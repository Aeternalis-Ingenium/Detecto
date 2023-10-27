from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto


class OTPDetecto(Detecto):
    def fit(self, dataset: DataFrame | list) -> None:  # type: ignore
        pass

    def score(self, dataset: DataFrame | list) -> DataFrame:  # type: ignore
        pass

    def predict(self, dataset: DataFrame | list) -> DataFrame:  # type: ignore
        pass

    def evaluate(self, dataset: DataFrame | list, prediction: DataFrame | list) -> DataFrame:  # type: ignore
        pass

    def set_params(self, **params: dict) -> None:  # type: ignore
        pass

    def get_params(self) -> dict:  # type: ignore
        pass
