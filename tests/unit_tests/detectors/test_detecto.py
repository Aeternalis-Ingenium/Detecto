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
        self.assertIsNone(obj=self.detector.fit())

    def test_detect(self):
        self.assertIsNone(obj=self.detector.detect())

    def test_evaluate(self):
        self.assertIsNone(obj=self.detector.evaluate(dataset=self.test_df))

    def test_set_params(self):
        self.assertIsNone(obj=self.detector.set_params())  # type: ignore

    def test_get_params(self):
        self.assertIsNone(obj=self.detector.params)

    def tearDown(self) -> None:
        return super().tearDown()
