from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.interface import Timeframe


class AbstractDetectoTestModel(Detecto):
    def fit(self, dataset: DataFrame | list) -> None:
        pass

    def score(self, dataset: DataFrame | list) -> DataFrame:
        pass

    def detect(self, dataset: DataFrame | list) -> DataFrame:
        pass

    def evaluate(self, dataset: DataFrame | list, detected: DataFrame | list) -> DataFrame:
        pass

    def set_params(self, **params: dict) -> None:
        pass

    def get_params(self) -> dict:
        return {"param_1": "Test Param 1", "param_2": "Test Param 2"}


class AbstractTimeframeTestModel(Timeframe):
    def set_interval(self, **params: dict[str, int | float | bool]) -> None:
        pass
