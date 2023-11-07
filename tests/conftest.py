from pandas import DataFrame, read_csv

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.interface import Timeframe


class AbstractDetectoTestModel(Detecto):
    def fit(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        pass

    def detect(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        pass

    def evaluate(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        pass

    def set_params(self, **kwargs: int | float | str | None) -> None:
        pass

    @property
    def get_params(self) -> dict:
        return {"param_1": "Test Param 1", "param_2": "Test Param 2"}


class AbstractTimeframeTestModel(Timeframe):
    def set_interval(self, **kwargs: int | float | bool) -> None:
        pass


def init_df(path_to_file: str, shuffle: bool = True):
    df = read_csv(path_to_file)
    if shuffle:
        return df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df
