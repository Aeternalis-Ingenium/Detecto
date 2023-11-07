from pandas import DataFrame

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.pot import POTTimeframe


class POTDetecto(Detecto):
    """
    POTDetecto class implements the Peaks Over Threshold (POT) approach for anomaly detection.

    # Attributes
        * timeframe (POTTimeframe): Instance managing the timeframe details for the POT method.
        * __params (dict): Private dictionary to store parameters after model fitting.
    """

    def __init__(self):
        self.timeframe = POTTimeframe()
        self.__params = {}

    def compute_exceedance_threshold(self, dataset: DataFrame, q: float = 0.99) -> DataFrame:
        """
        Calculate the exceedance threshold for each feature in the dataset.

        # Parameters
            * dataset (DataFrame): The dataset to calculate the threshold for.
            * q (float): The quantile to use for thresholding.

        # Returns
            * DataFrame: The threshold values for each feature.
        """
        return dataset.expanding(min_periods=self.timeframe.t0).quantile(q=q).bfill()

    def extract_exceedance(
        self,
        dataset: DataFrame,
        exceedance_threshold_dataset: DataFrame,
        fill_value: float | None = 0.0,
        clip_lower: float | None = 0.0,
    ) -> DataFrame:
        """
        Extract values from the dataset that exceed the threshold values.

        # Parameters
            * dataset (DataFrame): The original dataset to compare against thresholds.
            * exceedance_threshold_dataset (DataFrame): Calculated thresholds for the dataset.
            * fill_value (float | None): Value to fill missing entries with before comparison.
            * clip_lower (float | None): Minimum value to clip data to after subtraction.

        # Returns
            * DataFrame: The dataset with values exceeding the thresholds.
        """
        return dataset.subtract(exceedance_threshold_dataset, fill_value=fill_value).clip(lower=clip_lower)

    def fit(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        pass

    def compute_anomaly_threshold(self, dataset: DataFrame):
        pass

    def detect(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        pass

    def evaluate(self, dataset: DataFrame, **kwargs: DataFrame | list | int | float | None) -> DataFrame:
        pass

    def set_params(self, **kwargs: int | float | str | None) -> None:
        pass

    @property
    def get_params(self) -> dict:  # type: ignore
        pass

    def __str__(self):
        return "Peak Over Threshold Anomaly Detector"
