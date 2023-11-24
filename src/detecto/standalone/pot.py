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
