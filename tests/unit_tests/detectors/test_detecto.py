from unittest import TestCase

from pandas import DataFrame

from tests.conftest import AbstractDetectoTestModel


class TestDetectoInterfaceClass(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.detector = AbstractDetectoTestModel()

    def test_fit(self):
        self.assertIsNone(obj=self.detector.fit())

    def test_detect(self):
        self.assertIsNone(obj=self.detector.detect())

    def test_evaluate(self):
        self.assertIsNone(obj=self.detector.evaluate())

    def test_set_params(self):
        self.assertIsNone(obj=self.detector.set_params())  # type: ignore

    def test_get_params(self):
        self.assertIsNone(obj=self.detector.params)

    def tearDown(self) -> None:
        return super().tearDown()
