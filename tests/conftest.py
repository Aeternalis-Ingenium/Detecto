from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto


class AbstractDetectoTestModel(Detecto):
    def fit(self, dataset: DataFrame | list) -> None:
        pass

    def score(self, dataset: DataFrame | list) -> DataFrame:
        pass

    def predict(self, dataset: DataFrame | list) -> DataFrame:
        pass

    def evaluate(self, dataset: DataFrame | list, prediction: DataFrame | list) -> DataFrame:
        pass

    def set_params(self, **params: dict) -> None:
        pass

    def get_params(self) -> dict:
        return {"param_1": "Test Param 1", "param_2": "Test Param 2"}
