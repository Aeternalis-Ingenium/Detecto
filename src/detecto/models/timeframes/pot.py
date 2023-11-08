from src.detecto.models.timeframes.interface import Timeframe


class POTTimeframe(Timeframe):
    """
    POTTimeframe manages the intervals used in the Peaks Over Threshold (POT) method for anomaly detection.

    Attributes:
        * t0 (float | None): The starting point of the timeframe for learning the longest behavior of the data.
        * t1 (float | None): The starting point of the timeframe for setting the threshold.
        * t2 (float | None): The starting point of the timeframe for detecting anomalies.
    """

    def __init__(self) -> None:
        self.t0: int | None = None
        self.t1: int | None = None
        self.t2: int | None = None

    def set_interval(self, **kwargs: int | float | bool) -> None:  # type: ignore
        """
        Sets the intervals t0, t1, and t2 based on the dataset size and the specified percentages of the dataset
        to be used for learning, threshold setting, and anomaly detection.

        # Parameters
            * #### kwargs
                * total_rows (int): The total number of rows in the dataset.
                * t0_percentage (float): The percentage of the total rows to use for learning the normal behavior.
                * t1_percentage (float): The percentage of the total rows to use for setting the threshold.
                * t2_percentage (float): The percentage of the total rows to use for detecting anomalies.
                * prod_mode (bool): A flag indicating whether the model is in production mode, which affects the interval calculations.
        """
        total_rows = kwargs.get("total_rows")
        t0_percentage = kwargs.get("t0_percentage", 0.6)
        t1_percentage = kwargs.get("t1_percentage", 0.25)
        t2_percentage = kwargs.get("t2_percentage", 0.15)
        prod_mode = kwargs.get("prod_mode", False)

        if prod_mode:
            t2 = 1
            t0_percentage = 0.6
            t1_percentage = 0.4
            total_rows = total_rows - t2  # type: ignore
            t0 = int(t0_percentage * total_rows)  # type: ignore
            t1 = int(t1_percentage * total_rows)  # type: ignore
            uncounted_days = t0 + t1 - total_rows  # type: ignore
        else:
            t0 = int(t0_percentage * total_rows)  # type: ignore
            t1 = int(t1_percentage * total_rows)  # type: ignore
            t2 = int(t2_percentage * total_rows)  # type: ignore
            uncounted_days = t0 + t1 + t2 - total_rows  # type: ignore

        if uncounted_days < 0:
            t1 += abs(uncounted_days)  # type: ignore
        elif uncounted_days > 0:
            t1 -= uncounted_days  # type: ignore

        self.t0, self.t1, self.t2 = t0, t1, t2

    def __str__(self):
        return "Peak Over Threshold Timeframe"
