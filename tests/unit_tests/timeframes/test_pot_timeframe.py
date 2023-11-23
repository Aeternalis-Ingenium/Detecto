from unittest import TestCase

from src.detecto.models.timeframes.interface import Timeframe
from src.detecto.models.timeframes.pot import POTTimeframe


class TestPOTDetecto(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.timeframe = POTTimeframe()

    def test_instance_abstract_class(self):
        self.assertIsInstance(obj=self.timeframe, cls=Timeframe)

    def test_string_method(self):
        self.assertTrue(expr=str(self.timeframe) == "Peak Over Threshold Timeframe")

    def test_construct_pot_detecto_successful(self):
        self.assertIsNone(obj=self.timeframe.t0)
        self.assertIsNone(obj=self.timeframe.t1)
        self.assertIsNone(obj=self.timeframe.t2)

    def test_pot_detecto_set_time_windows_method_dev_mode(self):
        self.timeframe.set_interval(total_rows=10000)

        self.assertTrue(expr=self.timeframe.t0 == 6000)
        self.assertTrue(expr=self.timeframe.t1 == 2500)
        self.assertTrue(expr=self.timeframe.t2 == 1500)

    def test_pot_detecto_set_time_windows_method_prod_mode(self):
        self.timeframe.set_interval(total_rows=10000, prod_mode=True)

        self.assertTrue(expr=self.timeframe.t0 == 5999)
        self.assertTrue(expr=self.timeframe.t1 == 4000)
        self.assertTrue(expr=self.timeframe.t2 == 1)
