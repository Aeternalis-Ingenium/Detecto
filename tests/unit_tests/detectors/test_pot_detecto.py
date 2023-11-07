from unittest import TestCase

from pandas import DataFrame, testing as pd_testing
from scipy.stats import genpareto

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
        self.df_2 = DataFrame(
            {
                "df_2_feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "df_2_feature_2": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
            }
        )
        self.detector.timeframe.set_interval(
            total_rows=self.df_1.shape[0],
        )

    def test_construct_pot_detecto_successful(self):
        expected_detected_df = DataFrame(
            data={
                "is_anomaly_col_1": [False, False, False, False, False, False, False, False, True, True],
                "is_anomaly_col_2": [False, False, False, False, True, False, False, False, False, False],
            }
        )

        assert isinstance(self.detector, Detecto)
        assert str(self.detector) == "Peak Over Threshold Anomaly Detector"
        self.assertIsNone(self.detector.evaluate(dataset=self.df_1, detected=expected_detected_df))

    def test_compute_exceedance_threshold(self):
        exceedance_threshold_df = self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.99)

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

    def test_extract_exeedances(self):
        expected_threshold_df = DataFrame(
            data={
                "df_1_feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "df_1_feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        exceedance_threshold_df = self.detector.compute_exceedance_threshold(dataset=self.df_1, q=0.99)

        pd_testing.assert_frame_equal(left=exceedance_threshold_df, right=expected_threshold_df)

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
        exceedance_df = self.detector.extract_exceedance(
            dataset=self.df_1, exceedance_threshold_dataset=exceedance_threshold_df, fill_value=0.0, clip_lower=0.0
        )

        pd_testing.assert_frame_equal(left=exceedance_df, right=expected_exceedance_df)
        pd_testing.assert_series_equal(
            left=exceedance_df["df_1_feature_1"], right=expected_exceedance_df["df_1_feature_1"]
        )
        pd_testing.assert_series_equal(
            left=exceedance_df["df_1_feature_2"], right=expected_exceedance_df["df_1_feature_2"]
        )

    def test_genpareto_fitting_method(self):
        pass

    def test_detect_method(self):
        pass

    def test_evaluate_method(self):
        pass

    def test_set_params_method(self):
        pass

    def test_get_params_method(self):
        pass

    def tearDown(self) -> None:
        return super().tearDown()
