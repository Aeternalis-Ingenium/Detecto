from numpy import quantile
from pandas import DataFrame
from scipy.stats import genpareto


def compute_pot_threshold(dataset: DataFrame, t0: int, q: float = 0.99) -> DataFrame:
    """
    Calculate the exceedance threshold for each feature in the dataset.

    # Parameters
        * dataset (DataFrame): The dataset to calculate the threshold for.
        * t0 (int): The minimum timeframe of observation to have a value, otherwise `np.NaN`.
        * q (float): The quantile to use for thresholding.

    # Returns
        * DataFrame: The threshold for each feature.
    """
    return dataset.expanding(min_periods=t0).quantile(q=q).bfill()


def extract_pot_data(
    dataset: DataFrame,
    pot_threshold_dataset: DataFrame,
    fill_value: float | None = 0.0,
    clip_lower: float | None = 0.0,
) -> DataFrame:
    """
    Extract values from the dataset that exceed the threshold values.

    # Parameters
        * dataset (DataFrame): The original dataset to compare against thresholds.
        * pot_threshold_dataset (DataFrame): The DataFrame with POT thresholds to compute the exceedances.
        * fill_value (float | None): Value to fill missing entries with before comparison.
        * clip_lower (float | None): Minimum value to clip data to after subtraction.

    # Returns
        * DataFrame: The dataset with values exceeding the thresholds.
    """
    return dataset.subtract(pot_threshold_dataset, fill_value=fill_value).clip(lower=clip_lower)


def __set_gpd_params_structure(total_rows: int) -> dict[int, list]:
    """
    Initialize the parameter structure for storing model parameters.

    # Parameters
        * total_rows (list[int]): The total number of row index of a DataFrame.

    # Returns
        * dict[int, list]: The structure of parameter results from the fitting function.
    """
    params: dict = {}

    for row in range(0, total_rows):
        params[row] = []

    return params


def set_gpd_params(
    params: dict[int, list],
    feature_name: str,
    row: int,
    anomaly_score: float,
    c: float | None = None,
    loc: float | None = None,
    scale: float | None = None,
    p_value: float | None = None,
) -> None:
    """
    Set the parameters obtained after fitting the model.

    # Parameters
    * feature_name (str): The name of the feature (column) to store the parameters and statistics of fitting result.
    * row (int): The number of row that points the index of the data point.
    * c (float | None): This parameter determines the tail behavior of the distribution: > 0 (heavy), == 0 (exponential distribution), < 0 (finite endpoint).
    * loc (float | None): The parameter that shifts the distribution along the horizontal axis and typically marks the threshold above which the tail begins.
    * scale (float | None): The parameter that stretches or shrinks the distribution along the horizontal axis: Larger value shows the spread out of extreme values.
    * p_value (float | None): The result from calculating the survival function, 1 - CDF, that determines the probability of observing an extreme value.
    * anomaly_score (float | None): The inverted p-value (1 / p-value) for each cell or The accumulated inverted p-values per row, depending the feature name.

    # Example
    A dataset with two columns and two rows:
    ```json
        {
            0: [
                {
                    'col_1': {
                        'gpd_params': {
                            'c': 0.0,
                            'loc': 0.0,
                            'scale': 0.0
                            },
                        'gpd_stats': {
                            'anomaly_score': 0.0,
                            'p_value': 0.0
                            }
                    }
                },
                {
                    'col_2': {
                        'gpd_params': {
                            'c': 0.0,
                            'loc': 0.0,
                            'scale': 0.0
                            },
                        'gpd_stats': {
                            'anomaly_score': 0.0,
                            'p_value': 0.0
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
                            'loc': 0.02,
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
                            'c': 0.0,
                            'loc': 0.0,
                            'scale': 0.0
                            },
                        'gpd_stats': {
                            'anomaly_score': 0.0,
                            'p_value': 0.0
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
        * dict[int, list[dict[str, dict[str, float] | float]]]: GPD parameters and statistics from fitting into Gen. Pareto.
    """
    data = (
        {"total_anomaly_score": anomaly_score}
        if feature_name == "total_anomaly_score"
        else {
            feature_name: {
                "gpd_params": {
                    "c": c,  # type: ignore
                    "loc": loc,
                    "scale": scale,
                },
                "gpd_stats": {
                    "p_value": p_value,
                    "anomaly_score": anomaly_score,
                },
            }
        }
    )
    params[row].append(data)


def fit_pot_data(
    dataset: DataFrame, pot_dataset: DataFrame, t0: int
) -> tuple[dict[int, list[dict[str, dict[str, float] | float]]], DataFrame]:
    """
    Fit the POT model on the dataset and calculate anomaly scores for each feature.

    # Parameters
        * dataset (DataFrame): The dataset on which the POT model is to be fitted.
        * kwargs:
            * exceedance_dataset (DataFrame): The dataset containing exceedance values.

    # Returns
        * DataFrame: Anomaly scores for each feature in the dataset.
    """
    anomaly_scores = dataset.drop(dataset.index).add_prefix("anomaly_score_").to_dict(orient="list")
    anomaly_scores["total_anomaly_score"] = []
    t1_t2_pot_data = pot_dataset.iloc[t0:]  # type: ignore

    gpd_params = __set_gpd_params_structure(total_rows=t1_t2_pot_data.shape[0])

    for row in range(0, t1_t2_pot_data.shape[0]):
        pot_data_for_learning = pot_dataset.iloc[: t0 + row]  # type: ignore
        pot_data_of_interest = t1_t2_pot_data.iloc[[row]]
        total_anomaly_score_per_row = 0.0

        for feature_name in t1_t2_pot_data.columns:
            pot_data_for_fitting: list[float | None] = pot_data_for_learning[feature_name][
                pot_data_for_learning[feature_name] > 0.0
            ].to_list()
            if pot_data_of_interest[feature_name].iloc[0] > 0:
                if len(pot_data_for_fitting) > 0:
                    (c, loc, scale) = genpareto.fit(data=pot_data_for_fitting, floc=0)
                    p_value: float = genpareto.sf(
                        x=pot_data_of_interest[feature_name].iloc[0], c=c, loc=loc, scale=scale
                    )
                    inverted_p_value = 1 / p_value if p_value > 0.0 else float("inf")
                    total_anomaly_score_per_row += inverted_p_value
                    set_gpd_params(
                        params=gpd_params,
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
                    set_gpd_params(
                        params=gpd_params,
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
                set_gpd_params(
                    params=gpd_params,
                    feature_name=feature_name,
                    row=row,
                    c=0.0,
                    loc=0.0,
                    scale=0.0,
                    p_value=0.0,
                    anomaly_score=0.0,
                )
                anomaly_scores[f"anomaly_score_{feature_name}"].append(0.0)
        set_gpd_params(
            params=gpd_params,
            feature_name="total_anomaly_score",
            row=row,
            anomaly_score=total_anomaly_score_per_row,
        )
        anomaly_scores["total_anomaly_score"].append(total_anomaly_score_per_row)
    return (gpd_params, DataFrame(data=anomaly_scores))


def compute_extreme_anomaly_threshold(
    dataset: DataFrame, total_anomaly_score_column: str, t1: int, q: float = 0.80
) -> float:
    clean_dataset = dataset[
        (dataset[total_anomaly_score_column] > 0) & (dataset[total_anomaly_score_column] != float("inf"))
    ]

    if len(clean_dataset) == 0:
        raise ValueError("There are no total anomaly scores per row > 0")

    return quantile(
        a=clean_dataset.iloc[:t1][total_anomaly_score_column].to_list(),
        q=q,
    )


def detect_extreme_anomaly(
    dataset: DataFrame, total_anomaly_score_column: str, t1: int, extreme_anomaly_threshold: float
) -> DataFrame:
    anomaly_data = {}
    t2_dataset = dataset.iloc[t1:]
    anomaly_data["is_anomaly"] = (
        t2_dataset[total_anomaly_score_column].apply(lambda x: x > extreme_anomaly_threshold).to_list()
    )

    return DataFrame(data=anomaly_data)
