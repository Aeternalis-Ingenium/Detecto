from unittest import TestCase

from pandas import DataFrame, Series, testing

from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.detectors.pot import POTDetecto


class TestPOTDetecto(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.detector = POTDetecto()
        self.df = DataFrame(
            data={
                "feature_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "feature_2": [15, 17, 24, 36, 23, 15, 75, 56, 89, 105],
            }
        )

    def test_construct_pot_detecto_successful(self):
        assert isinstance(self.detector, Detecto)
        assert str(self.detector) == "Peak Over Threshold Anomaly Detector"
        self.assertIsNone(self.detector.fit(dataset=[1, 2, 3]))  # type: ignore
        self.assertIsNone(self.detector.score(dataset=[1, 2, 3]))
        self.assertIsNone(self.detector.detect(dataset=[1, 2, 3]))
        self.assertIsNone(self.detector.evaluate(dataset=[3000, 8, 24, 6000], detected=[1, 0, 0, 1]))
        self.assertIsNone(self.detector.set_params())  # type: ignore

    def test_pot_threshold_quantile_default_value(self):
        assert type(self.detector.pot_threshold_quantile) == float  # type: ignore
        assert self.detector.pot_threshold_quantile == 0.99  # type: ignore

    def test_anomaly_threshold_quantile_default_value(self):
        assert type(self.detector.anomaly_threshold_quantile) == float  # type: ignore
        assert self.detector.anomaly_threshold_quantile == 0.99  # type: ignore

    def test_compute_pot_threshold(self):
        self.detector.compute_pot_threshold(dataset=self.df)

        expected_pot_threshold = DataFrame(
            data={
                "feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        testing.assert_series_equal(
            left=self.detector.pot_threshold["feature_1"], right=expected_pot_threshold["feature_1"]  # type: ignore
        )
        testing.assert_series_equal(
            left=self.detector.pot_threshold["feature_2"], right=expected_pot_threshold["feature_2"]  # type: ignore
        )

    def test_extract_exeedances(self):
        expected_pot_threshold = DataFrame(
            data={
                "feature_1": [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 69.4, 79.3, 89.2, 99.1],
                "feature_2": [35.40, 35.40, 35.40, 35.40, 35.40, 35.40, 72.66, 73.67, 87.88, 103.56],
            }
        )

        self.detector.compute_pot_threshold(dataset=self.df)

        testing.assert_series_equal(
            left=self.detector.pot_threshold["feature_1"], right=expected_pot_threshold["feature_1"]  # type: ignore
        )
        testing.assert_series_equal(
            left=self.detector.pot_threshold["feature_2"], right=expected_pot_threshold["feature_2"]  # type: ignore
        )

        expected_exceedances = DataFrame(
            data={
                "feature_1": [
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
                "feature_2": [
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
        exceedances = self.detector.extract_exceedances(dataset=self.df)

        testing.assert_series_equal(left=exceedances["feature_1"], right=expected_exceedances["feature_1"])
        testing.assert_series_equal(left=exceedances["feature_2"], right=expected_exceedances["feature_2"])

    def test_genpareto_fitting_method(self):
        pass

    def test_score_method(self):
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
        pass
