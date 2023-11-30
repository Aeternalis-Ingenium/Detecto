from numpy import quantile
from pandas import DataFrame, Series
from scipy.stats import genpareto, ks_1samp

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.timeframes.pot import POTTimeframe


class POTDetecto(Detecto):
    """
    POTDetecto class implements the Peaks Over Threshold (POT) approach for anomaly detection.

    # Attributes
        * timeframe (POTTimeframe): A timeframe instance that manages the time windows (t0, t1, t2) for the "Peak over Threshold" method.
        * exceedance_threshold_dataset (DataFrame | None): A Pandas DataFrame that holds the threshold to get the exceedances, default is None.
        * exceedance_dataset (DataFrame | None): A Pandas DataFrame that holds the exceedances, default is None.
        * anomaly_score_dataset (DataFrame | None): A Pandas dataFrame the stores the anomaly scores from gen. pareto fitting, default is None.
        * anomaly_threshold (DataFrame | None): A single float that serves as the threshold to measure the anomalous data, default is None.
        * anomaly_dataset (DataFrame | None): A Pandas DataFrame that serves as the final dataset where anomalies are observable, default is None.
        * ktest_result (DataFrame | None): The evaluation result of the exceedances and GPD params distribution via Kolmogorov Smirnov test, default is None.
        * __params (dict[str, list[dict[int, dict[str, float | None]]]]): Private dictionary to store parameters after model fitting.
    """

    def __init__(self):
        self.timeframe = POTTimeframe()
        self.exceedance_threshold_dataset = None
        self.exceedance_dataset = None
        self.anomaly_score_dataset = None
        self.anomaly_threshold = None
        self.anomaly_dataset = None
        self.ktest_result = None
        self.__params = {}

    def __set_params_structure(self, total_rows: int) -> None:
        """
        Initialize the parameter structure for storing model parameters.

        # Parameters
            * total_rows (list[int]): The total number of row index from the dataset.

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
            * dict[str, list[dict[int, dict[str, float | None]]]]: A dictionary containing the GPD fit parameters and statistics for each row and feature.
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

    def __get_nonzero_params(self, feature_name: str) -> list[tuple[int, tuple[float, float, float]]]:
        if self.timeframe.t1 is None:
            raise ValueError("`timeframes` are not set yet. Need to call `timeframe.set_interval()` first!")

        if len(self.params) == 0:
            raise ValueError("`__params` is still empty. Need to call `fit()` first!")

        nonzero_params = []
        for row_index in range(0, self.timeframe.t1 + self.timeframe.t2):  # type: ignore
            for data_dict in self.params[row_index]:  # type: ignore
                for key in data_dict.keys():
                    if key == feature_name:
                        if (
                            data_dict[key]["gpd_params"]["c"] != 0  # type: ignore
                            or data_dict[key]["gpd_params"]["loc"] != 0  # type: ignore
                            or data_dict[key]["gpd_params"]["scale"] != 0  # type: ignore
                        ):
                            nonzero_params.append(
                                (
                                    row_index,
                                    (
                                        data_dict[key]["gpd_params"]["c"],  # type: ignore
                                        data_dict[key]["gpd_params"]["loc"],  # type: ignore
                                        data_dict[key]["gpd_params"]["scale"],  # type: ignore
                                    ),
                                )
                            )
        return nonzero_params

    def compute_exceedance_threshold(self, dataset: DataFrame, q: float = 0.99) -> None:
        """
        Calculate the exceedance threshold for each feature in the dataset.

        # Parameters
            * dataset (DataFrame): The dataset to calculate the threshold for.
            * q (float): The quantile to use for thresholding.

        # Returns
            * None: The result is a Pandas DataFrame with threshold values for each feature, assigned into `exceedance_threshold_dataset`.
        """
        if not isinstance(dataset, DataFrame):
            raise ValueError("The `dataset` parameter needs to be a Pandas DataFrame!")

        if self.timeframe.t0 is None:
            raise ValueError("The `t0` period is not set! Call `timeframe.set_interval()` first!")

        try:
            self.exceedance_threshold_dataset = dataset.expanding(min_periods=self.timeframe.t0).quantile(q=q).bfill()
        except Exception as e:
            print(e)
            raise

    def extract_exceedance(
        self,
        dataset: DataFrame,
        fill_value: float | None = 0.0,
        clip_lower: float | None = 0.0,
    ) -> None:
        """
        Extract values from the dataset that exceed the threshold values.

        # Parameters
            * dataset (DataFrame): The original dataset to compare against thresholds.
            * exceedance_threshold_dataset (DataFrame): Calculated thresholds for the dataset.
            * fill_value (float | None): Value to fill missing entries with before comparison.
            * clip_lower (float | None): Minimum value to clip data to after subtraction.

        # Returns
            * None: The result is a Pandas DataFrame with values exceeding the thresholds, assigned into `exceedance_dataset`.
        """
        if not isinstance(dataset, DataFrame):
            raise ValueError("The `dataset` parameter needs to be a Pandas DataFrame!")

        if self.exceedance_threshold_dataset is None:
            raise ValueError(
                "The `exceedance_threshold_dataset` is not set! Call `compute_exceedance_threshold()` first!"
            )

        try:
            self.exceedance_dataset = dataset.subtract(
                other=self.exceedance_threshold_dataset, fill_value=fill_value
            ).clip(lower=clip_lower)
        except Exception as e:
            print(e)
            raise

    def fit(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        """
        Fit the POT model on the dataset and calculate anomaly scores for each feature.

        # Parameters
            * kwargs:
                * dataset (DataFrame): The original timeseries dataset on which the POT model is to be fitted.

        # Returns
            * None: The result is a Pandas DataFrame with anomaly scores for each feature in the dataset, assigned into `anomaly_score_dataset`.
        """
        dataset: DataFrame = kwargs.get("dataset", None)

        if dataset is None:
            raise ValueError("The `dataset` parameter can't be None. Please assign your original dataset!")
        elif type(dataset) != DataFrame:
            raise ValueError("The `dataset` parameter needs to be a Pandas DataFrame!")

        anomaly_scores = dataset.drop(dataset.index).add_prefix("anomaly_score_").to_dict(orient="list")
        anomaly_scores["total_anomaly_score"] = []
        t1_t2_exceedances = self.exceedance_dataset.iloc[self.timeframe.t0 :]  # type: ignore

        self.__set_params_structure(total_rows=t1_t2_exceedances.shape[0])

        for row in range(0, t1_t2_exceedances.shape[0]):
            exceedances_for_learning = self.exceedance_dataset.iloc[: self.timeframe.t0 + row]  # type: ignore
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
        self.anomaly_score_dataset = DataFrame(data=anomaly_scores)

    def compute_anomaly_threshold(self, q: float = 0.80) -> None:
        """
        Claculate the anomaly threshold with quantile method to be used to detect the anomalies.

        # Parameters
            * kwargs:
                * q (float): The quantile to calculate the threshold, range values are 0.0 - 1.0.

        # Returns
            * None: The threshold for anomalous data, assigned into `anomaly_threshold`.
        """
        if self.anomaly_score_dataset is None:
            raise ValueError("`anomaly_score_dataset` is still None. Need to call `.fit()` first!")

        try:
            anomaly_scores = (
                self.anomaly_score_dataset[  # type: ignore
                    (self.anomaly_score_dataset["total_anomaly_score"] > 0)  # type: ignore
                    & (self.anomaly_score_dataset["total_anomaly_score"] != float("inf"))  # type: ignore
                ]
                .iloc[: self.timeframe.t1]["total_anomaly_score"]
                .to_list()
            )
        except Exception as e:
            print(e)
            raise

        if len(anomaly_scores) == 0:
            raise ValueError("There are no total anomaly scores per row > 0")

        self.anomaly_threshold = quantile(
            a=anomaly_scores,
            q=q,
        )

    def detect(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        """
        Claculate the anomaly threshold with quantile method to be used to detect the anomalies.

        # Parameters
            * kwargs:
                * None: No parameters needed for this method.

        # Returns
            * None: The result is a Pandas DataFrame with boolean values where `True` indicates an anomaly, assigned into `anomaly_dataset`.
        """
        if self.anomaly_score_dataset is None:
            raise ValueError("`anomaly_score_dataset` is still None. Need to call `.fit()` first!")

        if self.anomaly_threshold is None:
            raise ValueError("`anomaly_threshold` is not set yet. Need to call `compute_anomaly_threshold()` first!")

        anomaly_data = {}

        try:
            t2_dataset: DataFrame = self.anomaly_score_dataset.iloc[self.timeframe.t1 :]  # type: ignore
            anomaly_data["is_anomaly"] = (
                t2_dataset["total_anomaly_score"].apply(lambda x: x > self.anomaly_threshold).to_list()
            )
        except Exception as e:
            print(e)
            raise

        self.anomaly_dataset = DataFrame(data=anomaly_data)

    def __ks_1sample(self, nonzero_exceedance_dataset: list[Series], stat_distance_threshold: float = 0.05) -> None:
        """
        The wrapper method for "1 Sample Kolmogorov Smirnov" test using `scipy.stats.ks_1samp()`.

        # Parameters
            * nonzero_exceedance_dataset (list[Series]): A list of Pandas Series that are deconstructed from the `exceedance_dataset`.
            * stat_distance_threshold (float): This parameter is used as the threshold to reject or accept the h0 that the 2 distributions are identical.

        # Returns
            * None: The test result is a Pandas DataFrame, assigned to `kstest_result`.
        """
        kstest_results: dict = {}

        for feature_idx, feature_name in enumerate(self.exceedance_dataset.columns):  # type: ignore
            non_zero_params = self.__get_nonzero_params(feature_name=feature_name)
            kstest_results[feature_name] = {}
            ks_result = ks_1samp(
                x=nonzero_exceedance_dataset[feature_idx],
                cdf=genpareto.cdf,
                args=(non_zero_params[-1][1][0], non_zero_params[-1][1][1], non_zero_params[-1][1][2]),
            )
            kstest_results[feature_name]["total_exceedances"] = len(nonzero_exceedance_dataset[feature_idx])
            kstest_results[feature_name]["stat_distance"] = ks_result.statistic
            kstest_results[feature_name]["p_value"] = ks_result.pvalue
            kstest_results[feature_name]["is_identical"] = ks_result.statistic < stat_distance_threshold
            kstest_results[feature_name]["c"] = non_zero_params[-1][1][0]
            kstest_results[feature_name]["loc"] = non_zero_params[-1][1][1]
            kstest_results[feature_name]["scale"] = non_zero_params[-1][1][2]

        self.kstest_result = DataFrame(
            data={
                "feature": [feature_name for feature_name in kstest_results.keys()],
                "total_exceedances": [
                    kstest_results[feature_name]["total_exceedances"] for feature_name in kstest_results.keys()
                ],
                "stat_distance": [
                    kstest_results[feature_name]["stat_distance"] for feature_name in kstest_results.keys()
                ],
                "p_value": [kstest_results[feature_name]["p_value"] for feature_name in kstest_results.keys()],
                "c": [kstest_results[feature_name]["c"] for feature_name in kstest_results.keys()],
                "loc": [kstest_results[feature_name]["loc"] for feature_name in kstest_results.keys()],
                "scale": [kstest_results[feature_name]["scale"] for feature_name in kstest_results.keys()],
                "is_identical": [
                    kstest_results[feature_name]["is_identical"] for feature_name in kstest_results.keys()
                ],
            }
        )

    def evaluate(self, **kwargs: DataFrame | list | str | int | float | None) -> None:
        """
        Evaluate the correlation between the result of `genpareto.fit()` (params) and the `exceedance_dataset`.

        # Parameters
            * kwargs:
                * method (Literal["ks", "qq"]):
                    * "ks": 1 sample "Kolmogorov Smirnov" test evaluates the statistical distance between two distributions.
                    * "qq": The "Quantile-Quantile" plot evaluates visually the linear correlation between the sample and theoretical quantiles.
                * stat_distance_threshold (float): This parameter only utilised when using Kolmogorov Smirnov test to reject or accept the h0.

        # Returns
            * None: The test result is either a Pandas DataFrame assigned to `kstest_result` or a plot to observe the visual correlation if `method = "qq"`.
        """
        if self.exceedance_dataset is None:
            raise ValueError("`exceedance_dataset` is still None. Need to call `extract_exceedance()` first!")

        filtered_exceedances_by_feature = [
            self.exceedance_dataset[self.exceedance_dataset[feature_name] > 0][feature_name].copy()
            for feature_name in self.exceedance_dataset.columns
        ]

        if kwargs.get("method") == "ks":
            stat_distance_threshold = kwargs.get("stat_distance_threshold", 0.03)
            self.__ks_1sample(
                nonzero_exceedance_dataset=filtered_exceedances_by_feature,
                stat_distance_threshold=stat_distance_threshold,  # type: ignore
            )

    def __str__(self):
        return "Peak Over Threshold Anomaly Detector"
