from unittest import TestCase

from tests.conftest import AbstractTimeframeTestModel


class TestTimeframeInterfaceClass(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.timeframe = AbstractTimeframeTestModel()

    def test_set_interval_method_dev_mode(self):
        self.assertIsNone(
            obj=self.timeframe.set_interval(
                total_rows=1000, t0_percentage=0.65, t1_percentage=0.25, t2_percentage=0.1, prod_mode=False  # type: ignore
            )
        )

    def test_set_interval_method_prod_mode(self):
        self.assertIsNone(
            obj=self.timeframe.set_interval(total_rows=10000, t0_percentage=0.60, t1_percentage=0.40, prod_mode=True)  # type: ignore
        )

    def tearDown(self) -> None:
        return super().tearDown()
