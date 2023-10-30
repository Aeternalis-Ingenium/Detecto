from unittest import TestCase

from tests.conftest import AbstractDetectoTestModel


class TestDetectoInterfaceClass(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.detector = AbstractDetectoTestModel()

    def test_fit(self):
        self.assertIsNone(self.detector.fit(dataset=[1, 2, 3]))  # type: ignore

    def test_score(self):
        self.assertIsNone(self.detector.score(dataset=[1, 2, 3]))

    def test_detect(self):
        self.assertIsNone(self.detector.detect(dataset=[1, 2, 3]))

    def test_evaluate(self):
        self.assertIsNone(self.detector.evaluate(dataset=[3000, 8, 24, 6000], detected=[1, 0, 0, 1]))

    def test_set_params(self):
        self.assertIsNone(self.detector.set_params())  # type: ignore

    def test_get_params(self):
        expected_params = {"param_1": "Test Param 1", "param_2": "Test Param 2"}

        assert self.detector.get_params() == expected_params

    def tearDown(self) -> None:
        return super().tearDown()
