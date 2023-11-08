from pandas import DataFrame, read_csv

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.interface import Timeframe


class AbstractDetectoTestModel(Detecto):
    def fit(self, dataset: DataFrame, **kwargs: DataFrame | list | str | int | float | None) -> DataFrame:
        pass

    def detect(self, dataset: DataFrame, **kwargs: DataFrame | list | str | int | float | None) -> DataFrame:
        pass

    def evaluate(self, dataset: DataFrame, **kwargs: DataFrame | list | str | int | float | None) -> DataFrame:
        pass

    @property
    def params(self) -> dict:  # type: ignore
        pass

    def set_params(self, **kwargs: str | int | float | None) -> None:
        pass


class AbstractTimeframeTestModel(Timeframe):
    def set_interval(self, **kwargs: int | float | bool) -> None:
        pass


def init_df(path_to_file: str, shuffle: bool = True):
    df = read_csv(path_to_file)
    if shuffle:
        return df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df
