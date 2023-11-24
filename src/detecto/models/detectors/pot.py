from numpy import quantile
from pandas import DataFrame
from scipy.stats import genpareto

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.pot import POTTimeframe


class POTDetecto(Detecto):
    """
    POTDetecto class implements the Peaks Over Threshold (POT) approach for anomaly detection.

    # Attributes
        * timeframe (POTTimeframe): Instance managing the timeframe details for the POT method.
        * anomaly_threshold (float): The threshold to measure the anomalous data.
        * __params (dict): Private dictionary to store parameters after model fitting.
    """

    def __init__(self):
        self.timeframe = POTTimeframe()
        self.anomaly_threshold = None
        self.__params = {}

    def __set_params_structure(self, total_rows: int) -> None:
        """
        Initialize the parameter structure for storing model parameters.

        # Parameters
            * feature_names (list[int]): A registry of GPD parameters and statistics per row index and feature.

        # Returns
            * None: Create structure assigned to `__params` attribute.
        """
        for row in range(0, total_rows):
            self.__params[row] = []

    @property
    def params(self) -> dict[str, list[dict[int, dict[str, float | None]]]]:
        """
        Get the parameters set from model fitting.

        # Returns
            * dict[Str, list[dict[int, dict[str, float | None]]]]: A dictionary containing the GPD fit parameters and statistics for each row and feature.
        """
        return self.__params

    def set_params(self, **kwargs: str | int | float | None) -> None:  # type: ignore
        """
        Set the parameters obtained after fitting the model.

        # Parameters
            * #### kwargs:
                * feature_name (str): The name of the feature (column) to store the parameters and statistics of fitting result.
                * row (int): The number of row that points the index of the data point.
                * c (float | None): This parameter determines the tail behavior of the distribution: > 0 (heavy), == 0 (exponential distribution), < 0 (finite endpoint).
                * loc (float | None): The parameter that shifts the distribution along the horizontal axis and typically marks the threshold above which the tail begins.
                * scale (float | None): The parameter that stretches or shrinks the distribution along the horizontal axis: Larger value shows the spread out of extreme values.
                * p_value (float | None): The result from calculating the survival function, 1 - CDF, that determines the probability of observing an extreme value.
                * anomaly_score (float | None): The inverted p-value (1 / p-value) for each cell.
                * total_anomaly_score (float): The accumulated inverted p-values per row.

        # Example
        A dataset with two columns and two rows:
        ```json
            {
                0: [
                    {
                        'col_1': {
                            'gpd_params': {
                                'c': None,
                                'loc': None,
                                'scale': None
                                },
                            'gpd_stats': {
                                'anomaly_score': 0.0,
                                'p_value': None
                                }
                        }
                    },
                    {
                        'col_2': {
                            'gpd_params': {
                                'c': None,
                                'loc': None,
                                'scale': None
                                },
                            'gpd_stats': {
                                'anomaly_score': 0.0,
                                'p_value': None
                                }
                        }
                    },
                    {'total_anomaly_score': 0.0}
                ],
                1: [
                    {
                        'col_1': {
                            'gpd_params': {
                                'c': 5.942200541853957,
                                'loc': 0,
                                'scale': 76.50452706730246
                                },
                            'gpd_stats': {
                                'anomaly_score': 1.4000018488279755,
                                'p_value': 0.7142847710073806
                                }
                            }
                        },
                    {
                        'col_2': {
                            'gpd_params': {
                                'c': None,
                                'loc': None,
                                'scale': None
                                },
                            'gpd_stats': {
                                'anomaly_score': 0.0,
                                'p_value': None
                                }
                        }
                    },
                    {
                        'total_anomaly_score': 1.4000018488279755
                    }
                ],
            }
        ```

        # Returns
            * None: Add all GPD params and statistics and append it to `__params` attribute.
        """
        feature_name: str = kwargs.get("feature_name")  # type: ignore
        data = (
            {"total_anomaly_score": kwargs.get("total_anomaly_score_per_row")}
            if feature_name == "total_anomaly_score"
            else {
                feature_name: {  # type: ignore
                    "gpd_params": {
                        "c": kwargs.get("c", None),
                        "loc": kwargs.get("loc", None),
                        "scale": kwargs.get("scale", None),
                    },
                    "gpd_stats": {
                        "p_value": kwargs.get("p_value", None),
                        "anomaly_score": kwargs.get("anomaly_score", None),
                    },
                }
            }
        )
        self.__params[kwargs.get("row")].append(data)

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

    def fit(self, dataset: DataFrame, **kwargs: DataFrame | list | str | int | float | None) -> DataFrame:
        """
        Fit the POT model on the dataset and calculate anomaly scores for each feature.

        # Parameters
            * dataset (DataFrame): The dataset on which the POT model is to be fitted.
            * kwargs:
                * exceedance_dataset (DataFrame): The dataset containing exceedance values.

        # Returns
            * DataFrame: Anomaly scores for each feature in the dataset.
        """
        exceedance_dataset = kwargs.get("exceedance_dataset", None)
        anomaly_scores = dataset.drop(dataset.index).add_prefix("anomaly_score_").to_dict(orient="list")
        anomaly_scores["total_anomaly_score"] = []
        t1_t2_exceedances = exceedance_dataset.iloc[self.timeframe.t0 :]  # type: ignore

        self.__set_params_structure(total_rows=t1_t2_exceedances.shape[0])

        for row in range(0, t1_t2_exceedances.shape[0]):
            exceedances_for_learning = exceedance_dataset.iloc[: self.timeframe.t0 + row]  # type: ignore
            exceedances_of_interest = t1_t2_exceedances.iloc[[row]]
            total_anomaly_score_per_row = 0.0

            for feature_name in t1_t2_exceedances.columns:
                exceedances_for_fitting: list[float | None] = exceedances_for_learning[feature_name][
                    exceedances_for_learning[feature_name] > 0.0
                ].to_list()
                if exceedances_of_interest[feature_name].iloc[0] > 0:
                    if len(exceedances_for_fitting) > 0:
                        (c, loc, scale) = genpareto.fit(data=exceedances_for_fitting, floc=0)
                        p_value: float = genpareto.sf(
                            x=exceedances_of_interest[feature_name].iloc[0], c=c, loc=loc, scale=scale
                        )
                        inverted_p_value = 1 / p_value if p_value > 0.0 else float("inf")
                        total_anomaly_score_per_row += inverted_p_value
                        self.set_params(
                            feature_name=feature_name,
                            row=row,
                            c=c,
                            loc=loc,
                            scale=scale,
                            p_value=p_value,
                            anomaly_score=inverted_p_value,
                        )
                        anomaly_scores[f"anomaly_score_{feature_name}"].append(inverted_p_value)
                    else:
                        self.set_params(
                            feature_name=feature_name,
                            row=row,
                            c=0.0,
                            loc=0.0,
                            scale=0.0,
                            p_value=0.0,
                            anomaly_score=0.0,
                        )
                        anomaly_scores[f"anomaly_score_{feature_name}"].append(0.0)
                else:
                    self.set_params(
                        feature_name=feature_name,
                        row=row,
                        c=0.0,
                        loc=0.0,
                        scale=0.0,
                        p_value=0.0,
                        anomaly_score=0.0,
                    )
                    anomaly_scores[f"anomaly_score_{feature_name}"].append(0.0)
            self.set_params(
                feature_name="total_anomaly_score", row=row, total_anomaly_score_per_row=total_anomaly_score_per_row
            )
            anomaly_scores["total_anomaly_score"].append(total_anomaly_score_per_row)
        return DataFrame(data=anomaly_scores)

    def compute_anomaly_threshold(self, dataset: DataFrame, q: float = 0.80):
        clean_dataset = dataset[
            (dataset["total_anomaly_score"] > 0) & (dataset["total_anomaly_score"] != float("inf"))
        ]

        if len(clean_dataset) == 0:
            raise ValueError("There are no total anomaly scores per row > 0")

        self.anomaly_threshold = quantile(
            a=clean_dataset.iloc[: self.timeframe.t1]["total_anomaly_score"].to_list(),
            q=q,
        )

    def detect(self, dataset: DataFrame, **kwargs: DataFrame | list | str | int | float | None) -> DataFrame:
        anomaly_data = {}
        t2_dataset = dataset.iloc[self.timeframe.t1 :]
        anomaly_data["is_anomaly"] = (
            t2_dataset["total_anomaly_score"].apply(lambda x: x > self.anomaly_threshold).to_list()
        )

        return DataFrame(data=anomaly_data)

    def evaluate(self, dataset: DataFrame, **kwargs: DataFrame | list | str | int | float | None) -> DataFrame:
        pass

    def __str__(self):
        return "Peak Over Threshold Anomaly Detector"
