from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.pot import POTTimeframe


class POTDetecto(Detecto):
    def __init__(self):
        self.timeframe = POTTimeframe()
        self.pot_threshold_quantile = 0.99
        self.pot_threshold = None
        self.anomaly_threshold_quantile = 0.99
        self.anomaly_threshold = None
        self.params = {}

    def compute_pot_threshold(self, dataset: DataFrame) -> None:
        if not self.timeframe.t0:
            self.timeframe.set_interval(total_rows=dataset.shape[0])
        self.pot_threshold = (
            dataset.expanding(min_periods=self.timeframe.t0).quantile(q=self.pot_threshold_quantile).bfill()
        )

    def extract_exceedances(self, dataset: DataFrame) -> DataFrame:
        return dataset.subtract(self.pot_threshold).clip(lower=0)

    def fit(self, dataset: DataFrame | list) -> None:  # type: ignore
        pass

    def score(self, dataset: DataFrame | list) -> DataFrame:  # type: ignore
        pass

    def detect(self, dataset: DataFrame | list) -> DataFrame:  # type: ignore
        pass

    def evaluate(self, dataset: DataFrame | list, detected: DataFrame | list) -> DataFrame:  # type: ignore
        pass

    def set_params(self, **params: dict) -> None:  # type: ignore
        pass

    def get_params(self) -> dict:  # type: ignore
        pass

    def __str__(self):
        return "Peak Over Threshold Anomaly Detector"
