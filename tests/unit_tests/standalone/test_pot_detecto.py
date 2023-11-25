from unittest import TestCase

from pandas import DataFrame, testing as pd_testing

from src.detecto.standalone.pot_detecto import (
    compute_extreme_anomaly_threshold,
    compute_pot_threshold,
    detect_extreme_anomaly,
    extract_pot_data,
    fit_pot_data,
    set_gpd_params,
)


class TestStandalonePOTDetectoFunctions(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.df_1 = DataFrame(
            data={
                "df_1_feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "df_1_feature_2": [15, 17, 24, 36, 23, 15, 75, 56, 89, 105],
            }
        )
        self.t0 = 6
        self.t1 = 3
        self.t2 = 1

    def test_compute_pot_threshold_function(self):
        exceedance_threshold_df = compute_pot_threshold(dataset=self.df_1, t0=self.t0, q=0.99)

        expected_exceedance_threshold = DataFrame(
            data={
                "df_1_feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "df_1_feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        pd_testing.assert_frame_equal(left=exceedance_threshold_df, right=expected_exceedance_threshold)
        pd_testing.assert_series_equal(
            left=exceedance_threshold_df["df_1_feature_1"], right=expected_exceedance_threshold["df_1_feature_1"]  # type: ignore
        )
        pd_testing.assert_series_equal(
            left=exceedance_threshold_df["df_1_feature_2"], right=expected_exceedance_threshold["df_1_feature_2"]  # type: ignore
        )

    def test_extract_pot_data_function(self):
        expected_pot_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "df_1_feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        pot_threshold_df = compute_pot_threshold(dataset=self.df_1, t0=self.t0, q=0.99)

        pd_testing.assert_frame_equal(left=pot_threshold_df, right=expected_pot_threshold_df)

        expected_pot_data_df = DataFrame(
            data={
                "df_1_feature_1": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.5,
                    0.6000000000000085,
                    0.7000000000000028,
                    0.7999999999999972,
                    0.9000000000000057,
                ],
                "df_1_feature_2": [
                    0,
                    0,
                    0,
                    0.5999999999999943,
                    0,
                    0,
                    2.3400000000000176,
                    0,
                    1.1200000000000045,
                    1.4399999999999977,
                ],
            }
        )
        pot_data_df = extract_pot_data(
            dataset=self.df_1, pot_threshold_dataset=pot_threshold_df, fill_value=0.0, clip_lower=0.0
        )

        pd_testing.assert_frame_equal(left=pot_data_df, right=expected_pot_data_df)
        pd_testing.assert_series_equal(
            left=pot_data_df["df_1_feature_1"], right=expected_pot_data_df["df_1_feature_1"]
        )
        pd_testing.assert_series_equal(
            left=pot_data_df["df_1_feature_2"], right=expected_pot_data_df["df_1_feature_2"]
        )

    def test_fit_pot_data_function_with_90_quantile(self):
        expected_pot_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 64.0, 73.0, 82.0, 91.0],
                "df_1_feature_2": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 51.6, 61.7, 77.8, 90.6],
            }
        )
        expected_pot_data_df = DataFrame(
            data={
                "df_1_feature_1": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                "df_1_feature_2": [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 23.4, 0.0, 11.2, 14.4],
            }
        )
        expected_extreme_anomaly_score_df = DataFrame(
            data={
                "anomaly_score_df_1_feature_1": [float("inf"), float("inf"), float("inf"), float("inf")],
                "anomaly_score_df_1_feature_2": [float("inf"), 0.0, 1.2988597467759642, 2.129427676525411],
                "total_anomaly_score": [float("inf"), float("inf"), float("inf"), float("inf")],
            }
        )

        pot_threshold_df = compute_pot_threshold(dataset=self.df_1, t0=self.t0, q=0.90)

        pd_testing.assert_frame_equal(left=pot_threshold_df, right=expected_pot_threshold_df)

        pot_data_df = extract_pot_data(
            dataset=self.df_1, pot_threshold_dataset=pot_threshold_df, fill_value=0.0, clip_lower=0.0
        )

        pd_testing.assert_frame_equal(left=pot_data_df, right=expected_pot_data_df)

        gpd_params, extreme_anomaly_score_df = fit_pot_data(dataset=self.df_1, pot_dataset=pot_data_df, t0=self.t0)

        pd_testing.assert_frame_equal(left=extreme_anomaly_score_df, right=expected_extreme_anomaly_score_df)
        self.assertEqual(first=type(gpd_params), second=dict)

    def test_compute_extreme_anomaly_threshold_function(self):
        expected_extreme_anomaly_threshold = 2.04403430931313
        test_df = DataFrame(
            data={
                "df_1_feature_1": [
                    263,
                    275,
                    56,
                    308,
                    488,
                    211,
                    70,
                    42,
                    67,
                    472,
                    304,
                    297,
                    480,
                    227,
                    453,
                    342,
                    115,
                    115,
                    67,
                    295,
                    9,
                    228,
                    89,
                    225,
                    360,
                    367,
                    418,
                    124,
                    229,
                    12,
                    111,
                    341,
                    209,
                    374,
                    254,
                    322,
                    99,
                    166,
                    435,
                    481,
                    106,
                    438,
                    180,
                    33,
                    30,
                    330,
                    139,
                    17,
                    268,
                    204000,
                ],
                "df_1_feature_2": [
                    387,
                    525,
                    520,
                    79,
                    345000,
                    474,
                    336,
                    159,
                    164,
                    110,
                    308,
                    434,
                    197,
                    470,
                    411,
                    100,
                    33,
                    205,
                    229,
                    295,
                    599,
                    535,
                    364,
                    24,
                    476,
                    582,
                    7,
                    418,
                    33,
                    149,
                    303,
                    11,
                    359,
                    484,
                    339,
                    69,
                    267,
                    297,
                    19,
                    69,
                    56,
                    533,
                    225,
                    524,
                    596,
                    304,
                    181,
                    535,
                    369,
                    351,
                ],
            }
        )
        t0, t1, t2 = 30, 13, 7
        expected_pot_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    453.0,
                    449.50000000000006,
                    446.0,
                    442.5,
                    439.00000000000006,
                    435.5,
                    431.99999999999994,
                    428.50000000000017,
                    438.6,
                    454.90000000000003,
                    453.0,
                    451.5,
                    450.00000000000006,
                    448.50000000000006,
                    447.0,
                    445.5,
                    444.0,
                    442.50000000000006,
                    441.00000000000006,
                    454.90000000000003,
                ],
                "df_1_feature_2": [
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    535.0,
                    534.0,
                    533.0,
                    532.0,
                    531.0,
                    530.0,
                    529.0,
                    528.0,
                    527.0,
                    526.0,
                    525.0,
                    532.2,
                    531.4000000000001,
                    530.6,
                    534.2,
                    534.0,
                    533.8,
                    535.0,
                    535.0,
                    535.0,
                ],
            }
        )
        expected_pot_data_df = DataFrame(
            data={
                "df_1_feature_1": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    33.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    17.099999999999966,
                    0.0,
                    0.0,
                    25.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    26.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    203545.1,
                ],
                "df_1_feature_2": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    344460.3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    59.299999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    42.299999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.7999999999999545,
                    0.0,
                    0.0,
                    61.799999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
        expected_extreme_anomaly_score_df = DataFrame(
            data={
                "anomaly_score_df_1_feature_1": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5947881128257808,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                ],
                "anomaly_score_df_1_feature_2": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0092577188337257,
                    0.0,
                    0.0,
                    2.1563458584349675,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "total_anomaly_score": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5947881128257808,
                    0.0,
                    1.0092577188337257,
                    0.0,
                    0.0,
                    2.1563458584349675,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                ],
            }
        )

        pot_threshold_df = compute_pot_threshold(dataset=test_df, t0=t0, q=0.90)

        pd_testing.assert_frame_equal(left=pot_threshold_df, right=expected_pot_threshold_df)

        pot_data_df = extract_pot_data(
            dataset=test_df, pot_threshold_dataset=pot_threshold_df, fill_value=0.0, clip_lower=0.0
        )

        pd_testing.assert_frame_equal(left=pot_data_df, right=expected_pot_data_df)

        gpd_params, extreme_anomaly_score_df = fit_pot_data(dataset=test_df, pot_dataset=pot_data_df, t0=t0)

        pd_testing.assert_frame_equal(left=extreme_anomaly_score_df, right=expected_extreme_anomaly_score_df)
        self.assertEqual(first=type(gpd_params), second=dict)

        extreme_anomaly_threshold = compute_extreme_anomaly_threshold(
            dataset=extreme_anomaly_score_df, total_anomaly_score_column="total_anomaly_score", t1=t1, q=0.90
        )

        self.assertEqual(first=extreme_anomaly_threshold, second=expected_extreme_anomaly_threshold)

    def test_detect_method(self):
        expected_extreme_anomaly_df = DataFrame(
            data={
                "is_anomaly": [
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                ]
            }
        )
        expected_extreme_anomaly_threshold = 2.04403430931313
        test_df = DataFrame(
            data={
                "df_1_feature_1": [
                    263,
                    275,
                    56,
                    308,
                    488,
                    211,
                    70,
                    42,
                    67,
                    472,
                    304,
                    297,
                    480,
                    227,
                    453,
                    342,
                    115,
                    115,
                    67,
                    295,
                    9,
                    228,
                    89,
                    225,
                    360,
                    367,
                    418,
                    124,
                    229,
                    12,
                    111,
                    341,
                    209,
                    374,
                    254,
                    322,
                    99,
                    166,
                    435,
                    481,
                    106,
                    438,
                    180,
                    33,
                    30,
                    330,
                    139,
                    17,
                    268,
                    204000,
                ],
                "df_1_feature_2": [
                    387,
                    525,
                    520,
                    79,
                    345000,
                    474,
                    336,
                    159,
                    164,
                    110,
                    308,
                    434,
                    197,
                    470,
                    411,
                    100,
                    33,
                    205,
                    229,
                    295,
                    599,
                    535,
                    364,
                    24,
                    476,
                    582,
                    7,
                    418,
                    33,
                    149,
                    303,
                    11,
                    359,
                    484,
                    339,
                    69,
                    267,
                    297,
                    19,
                    69,
                    56,
                    533,
                    225,
                    524,
                    596,
                    304,
                    181,
                    535,
                    369,
                    351,
                ],
            }
        )
        t0, t1, t2 = 30, 13, 7
        expected_pot_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    454.90000000000003,
                    453.0,
                    449.50000000000006,
                    446.0,
                    442.5,
                    439.00000000000006,
                    435.5,
                    431.99999999999994,
                    428.50000000000017,
                    438.6,
                    454.90000000000003,
                    453.0,
                    451.5,
                    450.00000000000006,
                    448.50000000000006,
                    447.0,
                    445.5,
                    444.0,
                    442.50000000000006,
                    441.00000000000006,
                    454.90000000000003,
                ],
                "df_1_feature_2": [
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    539.7,
                    535.0,
                    534.0,
                    533.0,
                    532.0,
                    531.0,
                    530.0,
                    529.0,
                    528.0,
                    527.0,
                    526.0,
                    525.0,
                    532.2,
                    531.4000000000001,
                    530.6,
                    534.2,
                    534.0,
                    533.8,
                    535.0,
                    535.0,
                    535.0,
                ],
            }
        )
        expected_pot_data_df = DataFrame(
            data={
                "df_1_feature_1": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    33.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    17.099999999999966,
                    0.0,
                    0.0,
                    25.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    26.099999999999966,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    203545.1,
                ],
                "df_1_feature_2": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    344460.3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    59.299999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    42.299999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.7999999999999545,
                    0.0,
                    0.0,
                    61.799999999999955,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )
        expected_extreme_anomaly_score_df = DataFrame(
            data={
                "anomaly_score_df_1_feature_1": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5947881128257808,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                ],
                "anomaly_score_df_1_feature_2": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0092577188337257,
                    0.0,
                    0.0,
                    2.1563458584349675,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "total_anomaly_score": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.5947881128257808,
                    0.0,
                    1.0092577188337257,
                    0.0,
                    0.0,
                    2.1563458584349675,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                ],
            }
        )

        pot_threshold_df = compute_pot_threshold(dataset=test_df, t0=t0, q=0.90)

        pd_testing.assert_frame_equal(left=pot_threshold_df, right=expected_pot_threshold_df)

        pot_data_df = extract_pot_data(
            dataset=test_df, pot_threshold_dataset=pot_threshold_df, fill_value=0.0, clip_lower=0.0
        )

        pd_testing.assert_frame_equal(left=pot_data_df, right=expected_pot_data_df)

        gpd_params, extreme_anomaly_score_df = fit_pot_data(dataset=test_df, pot_dataset=pot_data_df, t0=t0)

        pd_testing.assert_frame_equal(left=extreme_anomaly_score_df, right=expected_extreme_anomaly_score_df)
        self.assertEqual(first=type(gpd_params), second=dict)

        extreme_anomaly_threshold = compute_extreme_anomaly_threshold(
            dataset=extreme_anomaly_score_df, total_anomaly_score_column="total_anomaly_score", t1=t1, q=0.90
        )

        self.assertEqual(first=extreme_anomaly_threshold, second=expected_extreme_anomaly_threshold)

        extreme_anomaly_df = detect_extreme_anomaly(
            dataset=extreme_anomaly_score_df,
            total_anomaly_score_column="total_anomaly_score",
            t1=t1,
            extreme_anomaly_threshold=extreme_anomaly_threshold,
        )

        pd_testing.assert_frame_equal(left=extreme_anomaly_df, right=expected_extreme_anomaly_df)

    def test_evaluate_method(self):
        pass

    def test_params_attributes(self):
        expected_gpd_params = {
            0: [
                {
                    "anomaly_score_col_1": {
                        "gpd_params": {
                            "c": 0.123,
                            "loc": 0.002,
                            "scale": 0.001,
                        },
                        "gpd_stats": {"p_value": 0.05, "anomaly_score": 20.0},
                    },
                },
                {
                    "anomaly_score_col_2": {
                        "gpd_params": {
                            "c": 0.253,
                            "loc": 0.005,
                            "scale": 0.03,
                        },
                        "gpd_stats": {"p_value": 0.005, "anomaly_score": 200.0},
                    },
                },
                {
                    "total_anomaly_score": 220.0,
                },
            ]
        }
        gpd_params: dict = {}
        gpd_params[0] = []

        set_gpd_params(
            params=gpd_params,
            feature_name="anomaly_score_col_1",
            row=0,
            c=0.123,
            loc=0.002,
            scale=0.001,
            p_value=0.05,
            anomaly_score=20.0,
        )

        set_gpd_params(
            params=gpd_params,
            feature_name="anomaly_score_col_2",
            row=0,
            c=0.253,
            loc=0.005,
            scale=0.03,
            p_value=0.005,
            anomaly_score=200.0,
        )

        set_gpd_params(params=gpd_params, feature_name="total_anomaly_score", row=0, anomaly_score=220.0)

        self.assertEqual(first=type(gpd_params), second=type(expected_gpd_params))
        self.assertEqual(first=len(gpd_params[0]), second=len(expected_gpd_params[0]))  # type: ignore
        self.assertEqual(first=gpd_params[0], second=expected_gpd_params[0])  # type: ignore

    def tearDown(self) -> None:
        return super().tearDown()
