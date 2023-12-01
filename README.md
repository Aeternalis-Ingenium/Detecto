<h1 align=center>Detecto</h1>

Your ultimate anomaly detection library!

Detecto is only possible because of the following technologies:

- [Python 3.12.0](https://www.python.org/)
- [Pandas 2.1.1](https://pandas.pydata.org/)
- [NumPy 1.26.0](https://numpy.org/)
- [SciPy 1.11.3](https://scipy.org/)
- [Matplotlib 3.8.2](https://matplotlib.org/)
- [Pytest-Cov 4.1.0.](https://pytest-cov.readthedocs.io/en/latest/)
- [Black 23.10.0](https://black.readthedocs.io/en/stable/)
- [Isort 5.12.0](https://pycqa.github.io/isort/)
- [MyPy 1.6.1](https://mypy.readthedocs.io/en/stable/)
- [Bandit 1.7.5](https://bandit.readthedocs.io/en/latest/)

## **Introduction - Detection Methods**

There are many statistical methods that can be used to detect anomalous data. This library aims to implement the complete methods of anomaly detection, hence it will grow over time. All methods that are currently available can be seen in this list:

- [ ] Autoencoders
- [ ] Block Maxima
- [ ] Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- [ ] Isolation Forest
- [ ] Median Absolute Deviation (MAD)
- [ ] One-Class Support Vector Machine (SVM)
- [x] Peak Over Threshold with Generalised Pareto Distribution (POT with GPD)
- [ ] Z-Score

## Contribution

Want to contribute? Just clone this repo and let's start developing!

## Example Application

This is an example application to detect extreme value anomalies using Peak Over Threshold detector. Remember! In the case of a production grade detection project, the day-of-interest is always the present day. Thus it is always be 1 or we call it `t2 = 1`.

```python
from detector import Detecto, read_csv

# The original dataset must be a time series 
timeseries_df = read_csv(path="PATH/TO/DATASET.csv")

dto = Detecto(method="pot")

# Total rows is Total Row - 1, T0 is 60% of Total Row, T1 is 40% of Total Row, and T2 is 1
t0, t1, t2 = dto.set_t0_t1(total_rows=timeseries_df.shape[0], t0=.6, t1=.4)

# Dataset of daily Peak over Threshold score
pot_data_df = dto.compute_pot_data(df=timeseries_df, features=timeseries.columns, max_periods=t0, quantile=.97)

# Dataset of daily anomaly score
daily_anomaly_score_df = dto.compute_anomaly_score(df=pot_data_df, t0=t0, t1=t1)

# The final dataset with detected anomaly
anomaly_dataset_df = dto.detect_anomaly(df=daily_anomaly_score_df, t1=t1, t1_quantile=.97)
```
