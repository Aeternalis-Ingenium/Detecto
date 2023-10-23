from abc import ABCMeta, abstractmethod

from pandas import DataFrame


class Detecto(metaclass=ABCMeta):
    @abstractmethod
    def compute_anomaly_score(self, dataset: DataFrame) -> DataFrame:
        """A function to calculate the anomaly score for an aggregated timeframe.

        Args:
            :dataset (DataFrame): Pandas DataFrame object.

        Returns:
            :AnomalyScoreDataset (DataFrame): Pandas DataFrame object with anomaly scores.
        """

    @abstractmethod
    def detect_anomaly(self, dataset: DataFrame) -> DataFrame:
        """A function to detect the anomalous data.

        Args:
            :dataset (DataFrame): Pandas DataFrame object that contains anomaly scores.

        Returns:
            :AnomalyDataset (DataFrame): Pandas DataFrame object with the final result.
        """
