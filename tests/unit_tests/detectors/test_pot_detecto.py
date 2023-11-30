from unittest import TestCase

from pandas import DataFrame, testing as pd_testing
from scipy.stats import genpareto, ks_1samp

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.detectors.pot import POTDetecto


class TestPOTDetecto(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.detector = POTDetecto()
        self.df_1 = DataFrame(
            data={
                "df_1_feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "df_1_feature_2": [15, 17, 24, 36, 23, 15, 75, 56, 89, 105],
            }
        )
        self.detector.timeframe.set_interval(
            total_rows=self.df_1.shape[0],
        )

    def test_instance_is_abstract_class(self):
        self.assertIsInstance(obj=self.detector, cls=Detecto)

    def test_string_method(self):
        self.assertEqual(first=str(self.detector), second="Peak Over Threshold Anomaly Detector")

    def test_compute_exceedance_threshold_method(self):
        self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.99)

        expected_exceedance_threshold = DataFrame(
            data={
                "df_1_feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "df_1_feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        pd_testing.assert_frame_equal(
            left=self.detector.exceedance_threshold_dataset, right=expected_exceedance_threshold
        )
        pd_testing.assert_series_equal(
            left=self.detector.exceedance_threshold_dataset["df_1_feature_1"], right=expected_exceedance_threshold["df_1_feature_1"]  # type: ignore
        )
        pd_testing.assert_series_equal(
            left=self.detector.exceedance_threshold_dataset["df_1_feature_2"], right=expected_exceedance_threshold["df_1_feature_2"]  # type: ignore
        )

    def test_compute_exceedance_threshold_method_catches_value_error(self):
        with self.assertRaises(expected_exception=ValueError):
            self.detector.compute_exceedance_threshold(dataset=[0, 1, 2, 3, 4])

        pot_detecto = POTDetecto()

        with self.assertRaises(expected_exception=ValueError):
            pot_detecto.compute_exceedance_threshold(dataset=self.df_1)

    def test_extract_exeedance_method(self):
        expected_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "df_1_feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.99)

        pd_testing.assert_frame_equal(left=self.detector.exceedance_threshold_dataset, right=expected_threshold_df)

        expected_exceedance_df = DataFrame(
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
        self.detector.extract_exceedance(
            dataset=self.df_1,
            fill_value=0.0,
            clip_lower=0.0,
        )

        pd_testing.assert_frame_equal(left=self.detector.exceedance_dataset, right=expected_exceedance_df)
        pd_testing.assert_series_equal(
            left=self.detector.exceedance_dataset["df_1_feature_1"], right=expected_exceedance_df["df_1_feature_1"]  # type: ignore
        )
        pd_testing.assert_series_equal(
            left=self.detector.exceedance_dataset["df_1_feature_2"], right=expected_exceedance_df["df_1_feature_2"]  # type: ignore
        )

    def test_extract_exceedance_method_catches_value_error(self):
        with self.assertRaises(expected_exception=ValueError):
            self.detector.extract_exceedance(dataset=[0, 1, 2, 3, 4])

        pot_detecto = POTDetecto()
        pot_detecto.timeframe.set_interval(total_rows=self.df_1.shape[0])

        with self.assertRaises(expected_exception=ValueError):
            pot_detecto.extract_exceedance(dataset=self.df_1)

    def test_genpareto_fitting_method_with_90_quantile(self):
        expected_exceedance_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 64.0, 73.0, 82.0, 91.0],
                "df_1_feature_2": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 51.6, 61.7, 77.8, 90.6],
            }
        )
        expected_exceedance_df = DataFrame(
            data={
                "df_1_feature_1": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                "df_1_feature_2": [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 23.4, 0.0, 11.2, 14.4],
            }
        )
        expected_anomaly_score_df = DataFrame(
            data={
                "anomaly_score_df_1_feature_1": [float("inf"), float("inf"), float("inf"), float("inf")],
                "anomaly_score_df_1_feature_2": [float("inf"), 0.0, 1.2988597467759642, 2.129427676525411],
                "total_anomaly_score": [float("inf"), float("inf"), float("inf"), float("inf")],
            }
        )

        self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.90)

        pd_testing.assert_frame_equal(
            left=self.detector.exceedance_threshold_dataset, right=expected_exceedance_threshold_df
        )

        self.detector.extract_exceedance(
            dataset=self.df_1,
            fill_value=0.0,
            clip_lower=0.0,
        )

        pd_testing.assert_frame_equal(left=self.detector.exceedance_dataset, right=expected_exceedance_df)

        self.detector.fit(dataset=self.df_1)

        pd_testing.assert_frame_equal(left=self.detector.anomaly_score_dataset, right=expected_anomaly_score_df)

        exceedance_feature_1_df = expected_exceedance_df["df_1_feature_1"].copy()
        exceedance_feature_2_df = expected_exceedance_df["df_1_feature_2"].copy()
        filtered_exceedance_feature_1_df = exceedance_feature_1_df[exceedance_feature_1_df > 0]
        filtered_exceedance_feature_2_df = exceedance_feature_2_df[exceedance_feature_2_df > 0]

        self.assertEqual(first=filtered_exceedance_feature_1_df.shape[0], second=5)
        self.assertEqual(first=filtered_exceedance_feature_2_df.shape[0], second=4)

        feature_1_nonzero_params = self.detector._POTDetecto__get_nonzero_params(feature_name="df_1_feature_1")  # type: ignore
        feature_2_nonzero_params = self.detector._POTDetecto__get_nonzero_params(feature_name="df_1_feature_2")  # type: ignore

        self.assertIsNotNone(obj=feature_1_nonzero_params)
        self.assertIsNotNone(obj=feature_2_nonzero_params)

        kstest_feature_1_result = ks_1samp(
            x=filtered_exceedance_feature_1_df,
            cdf=genpareto.cdf,
            args=(
                feature_1_nonzero_params[-1][1][0],
                feature_1_nonzero_params[-1][1][1],
                feature_1_nonzero_params[-1][1][2],
            ),
        )

        kstest_feature_2_result = ks_1samp(
            x=filtered_exceedance_feature_2_df,
            cdf=genpareto.cdf,
            args=(
                feature_2_nonzero_params[-1][1][0],
                feature_2_nonzero_params[-1][1][1],
                feature_2_nonzero_params[-1][1][2],
            ),
        )

        self.assertEqual(first=kstest_feature_1_result.statistic, second=0.3999983048907735)
        self.assertEqual(first=kstest_feature_1_result.pvalue, second=0.30880444800752727)
        self.assertEqual(first=kstest_feature_2_result.statistic, second=0.25)
        self.assertEqual(first=kstest_feature_2_result.pvalue, second=0.90625)

    def test_fit_method_catches_value_error(self):
        self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.90)
        self.detector.extract_exceedance(
            dataset=self.df_1,
            fill_value=0.0,
            clip_lower=0.0,
        )

        with self.assertRaises(expected_exception=ValueError):
            self.detector.fit()

    def test_compute_anomaly_threshold_method(self):
        expected_anomaly_threshold = 2.04403430931313
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
        self.detector.timeframe.set_interval(total_rows=test_df.shape[0])
        expected_exceedance_threshold_df = DataFrame(
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
        expected_exceedance_df = DataFrame(
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
        expected_anomaly_score_df = DataFrame(
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

        self.detector.compute_exceedance_threshold(dataset=test_df, q=0.90)

        pd_testing.assert_frame_equal(
            left=self.detector.exceedance_threshold_dataset, right=expected_exceedance_threshold_df
        )

        self.detector.extract_exceedance(
            dataset=test_df,
            fill_value=0.0,
            clip_lower=0.0,
        )

        pd_testing.assert_frame_equal(left=self.detector.exceedance_dataset, right=expected_exceedance_df)

        self.detector.fit(dataset=test_df)

        pd_testing.assert_frame_equal(left=self.detector.anomaly_score_dataset, right=expected_anomaly_score_df)

        self.detector.compute_anomaly_threshold(q=0.90)

        self.assertEqual(first=self.detector.anomaly_threshold, second=expected_anomaly_threshold)

    def test_compute_anomaly_threshold_method_catches_value_error(self):
        self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.90)
        self.detector.extract_exceedance(
            dataset=self.df_1,
            fill_value=0.0,
            clip_lower=0.0,
        )

        with self.assertRaises(expected_exception=ValueError):
            self.detector.compute_anomaly_threshold()

    def test_detect_method(self):
        expected_anomaly_df = DataFrame(
            data={
                "is_anomaly": [
                    False,
                    False,
                    False,
                    False,
                    True,
                ]
            }
        )
        expected_anomaly_threshold = 2.04403430931313
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
        self.detector.timeframe.set_interval(total_rows=test_df.shape[0])
        expected_exceedance_threshold_df = DataFrame(
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
        expected_exceedance_df = DataFrame(
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
        expected_anomaly_score_df = DataFrame(
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

        self.detector.compute_exceedance_threshold(dataset=test_df, q=0.90)

        pd_testing.assert_frame_equal(
            left=self.detector.exceedance_threshold_dataset, right=expected_exceedance_threshold_df
        )

        self.detector.extract_exceedance(
            dataset=test_df,
            fill_value=0.0,
            clip_lower=0.0,
        )

        pd_testing.assert_frame_equal(left=self.detector.exceedance_dataset, right=expected_exceedance_df)

        self.detector.fit(dataset=test_df)

        pd_testing.assert_frame_equal(left=self.detector.anomaly_score_dataset, right=expected_anomaly_score_df)

        self.detector.compute_anomaly_threshold(q=0.90)

        self.assertEqual(first=self.detector.anomaly_threshold, second=expected_anomaly_threshold)

        self.detector.detect()

        pd_testing.assert_frame_equal(left=self.detector.anomaly_dataset, right=expected_anomaly_df)

    def test_detect_anomaly_method_catches_value_error(self):
        self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.90)
        self.detector.extract_exceedance(
            dataset=self.df_1,
            fill_value=0.0,
            clip_lower=0.0,
        )

        with self.assertRaises(expected_exception=ValueError):
            self.detector.detect()

        self.detector.fit(dataset=self.df_1)

        with self.assertRaises(expected_exception=ValueError):
            self.detector.detect()

    def test_get_nonzero_params_method_catches_value_error(self):
        detector = POTDetecto()

        with self.assertRaises(expected_exception=ValueError):
            detector._POTDetecto__get_nonzero_params(feature_name="feature_1")  # type: ignore

        detector.timeframe.set_interval(total_rows=1000)

        with self.assertRaises(expected_exception=ValueError):
            detector._POTDetecto__get_nonzero_params(feature_name="feature_1")  # type: ignore

    def test_ks_1sample_method_catches_value_error(self):
        with self.assertRaises(expected_exception=ValueError):
            self.detector._POTDetecto__ks_1sample(  # type: ignore
                nonzero_exceedance_dataset=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], stat_distance_threshold=0.03
            )

    def test_ks_evaluate_method_catches_value_error(self):
        with self.assertRaises(expected_exception=ValueError):
            self.detector.evaluate(method="ks", stat_distance_threshold=0.05)

    def test_evaluate_method_with_ks(self):
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
        self.detector.timeframe.set_interval(total_rows=test_df.shape[0])
        expected_exceedance_threshold_df = DataFrame(
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
        expected_exceedance_df = DataFrame(
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
        expected_anomaly_score_df = DataFrame(
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
        self.detector.compute_exceedance_threshold(dataset=test_df, q=0.90)

        pd_testing.assert_frame_equal(
            left=self.detector.exceedance_threshold_dataset, right=expected_exceedance_threshold_df
        )

        self.detector.extract_exceedance(
            dataset=test_df,
            fill_value=0.0,
            clip_lower=0.0,
        )

        pd_testing.assert_frame_equal(left=self.detector.exceedance_dataset, right=expected_exceedance_df)

        self.detector.fit(dataset=test_df)

        pd_testing.assert_frame_equal(left=self.detector.anomaly_score_dataset, right=expected_anomaly_score_df)

        self.detector.evaluate(method="ks", stat_distance_threshold=0.5)

        expected_kstest_result = DataFrame(
            data={
                "feature": ["df_1_feature_1", "df_1_feature_2"],
                "total_exceedances": [5, 5],
                "stat_distance": [0.4, 0.3015702053031097],
                "p_value": [0.3087999999999995, 0.6578151672671728],
                "c": [-3.507231273303336, 5.13336319598767],
                "loc": [0, 0],
                "scale": [116.08935514634031, 6.263007768565435],
                "is_identical": [True, True],
            }
        )

        pd_testing.assert_frame_equal(left=self.detector.kstest_result, right=expected_kstest_result)

        self.detector.evaluate(method="ks", stat_distance_threshold=0.3)
        expected_kstest_result["is_identical"] = [False, False]

        pd_testing.assert_frame_equal(left=self.detector.kstest_result, right=expected_kstest_result)

    def test_params_attributes(self):
        expected_params = {
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
        self.detector.params[0] = []  # type: ignore

        self.assertEqual(first=type(self.detector.params), second=dict)
        self.assertEqual(first=len(self.detector.params[0]), second=0)  # type: ignore

        self.detector.set_params(
            feature_name="anomaly_score_col_1",
            row=0,
            c=0.123,
            loc=0.002,
            scale=0.001,
            p_value=0.05,
            anomaly_score=20.0,
        )

        self.detector.set_params(
            feature_name="anomaly_score_col_2",
            row=0,
            c=0.253,
            loc=0.005,
            scale=0.03,
            p_value=0.005,
            anomaly_score=200.0,
        )

        self.detector.set_params(feature_name="total_anomaly_score", row=0, total_anomaly_score_per_row=220.0)

        self.assertEqual(first=type(self.detector.params), second=type(expected_params))
        self.assertEqual(first=len(self.detector.params[0]), second=len(expected_params[0]))  # type: ignore
        self.assertEqual(first=self.detector.params[0], second=expected_params[0])  # type: ignore

    def tearDown(self) -> None:
        return super().tearDown()
