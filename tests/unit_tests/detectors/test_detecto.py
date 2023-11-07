from unittest import TestCase

from pandas import DataFrame

from tests.conftest import AbstractDetectoTestModel


class TestDetectoInterfaceClass(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.detector = AbstractDetectoTestModel()
        self.test_df = DataFrame(
            data={
                "col_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col_2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            }
        )

    def test_fit(self):
        self.assertIsNone(self.detector.fit(dataset=self.test_df))  # type: ignore

    def test_detect(self):
        self.assertIsNone(self.detector.detect(dataset=self.test_df))

    def test_evaluate(self):
        expected_detected_df = DataFrame(
            data={
                "is_anomaly_col_1": [False, False, False, False, False, False, False, False, True, True],
                "is_anomaly_col_2": [False, False, False, False, True, False, False, False, False, False],
            }
        )
        self.assertIsNone(self.detector.evaluate(dataset=self.test_df, detected=expected_detected_df))

    def test_set_params(self):
        self.assertTrue(self.detector.set_params())  # type: ignore

    def test_get_params(self):
        expected_params = {"param_1": "Test Param 1", "param_2": "Test Param 2"}

        assert self.detector.get_params == expected_params

    def tearDown(self) -> None:
        return super().tearDown()
