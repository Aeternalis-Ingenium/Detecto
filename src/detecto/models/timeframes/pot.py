from src.detecto.models.timeframes.interface import Timeframe


class POTTimeframe(Timeframe):
    """
    POTTimeframe manages the intervals used in the Peaks Over Threshold (POT) method for anomaly detection.

    Attributes:
    ------------
        * t0 (int | None): The starting point of the timeframe for learning the longest behavior of the data.
        * t1 (int | None): The starting point of the timeframe for setting the threshold.
        * t2 (int | None): The starting point of the timeframe for detecting anomalies.
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
        ------------
            * #### kwargs
                * total_rows (int): The total number of rows in the dataset.
                * t0_percentage (float | None): The percentage of the total rows to use for learning the normal behavior, default 0.6.
                * t1_percentage (float | None): The percentage of the total rows to use for setting the threshold, default 0.3.
                * t2_percentage (float | None): The percentage of the total rows to use for detecting anomalies, default 0.1.
                * prod_mode (bool): A flag indicating whether the model is in production mode, which affects the interval calculations, default False.

        # Returns
        ------------
            * None: Calculate the percentage of t0, t1, t2 and assign them to t0, t1, and t2.
        """
        total_rows = kwargs.get("total_rows")
        t0_percentage = kwargs.get("t0_percentage", 0.6)
        t1_percentage = kwargs.get("t1_percentage", 0.3)
        t2_percentage = kwargs.get("t2_percentage", 0.1)
        prod_mode = kwargs.get("prod_mode", False)

        if prod_mode:
            if ((t0_percentage + t1_percentage > 1.0 or t0_percentage + t1_percentage == 1.0)) and (
                (t0_percentage - t1_percentage != 0.0) or (t1_percentage - t0_percentage != 0.0)
            ):
                raise ValueError(
                    "If `prod_mode = True`, t2 always = 1. Hence `t0_percentage` + `t1_percentage` must equal to 1.0 (100%)."
                )

            t2 = 1
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
