from pandas import DataFrame


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
