from unittest import TestCase

from src.detecto.models.timeframes.interface import Timeframe
from src.detecto.models.timeframes.pot import POTTimeframe


class TestPOTDetecto(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.timeframe = POTTimeframe()

    def test_construct_pot_detecto_successful(self):
        assert isinstance(self.timeframe, Timeframe)
        assert str(self.timeframe) == "Peak Over Threshold Timeframe"
        self.assertIsNone(self.timeframe.t0)
        self.assertIsNone(self.timeframe.t1)
        self.assertIsNone(self.timeframe.t2)

    def test_pot_detecto_set_time_windows_method_dev_mode(self):
        self.timeframe.set_interval(total_rows=10000)

        assert self.timeframe.t0 == 6000
        assert self.timeframe.t1 == 2500
        assert self.timeframe.t2 == 1500

    def test_pot_detecto_set_time_windows_method_prod_mode(self):
        self.timeframe.set_interval(total_rows=10000, prod_mode=True)

        assert self.timeframe.t0 == 5999
        assert self.timeframe.t1 == 4000
        assert self.timeframe.t2 == 1
