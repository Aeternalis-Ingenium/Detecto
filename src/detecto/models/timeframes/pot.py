from src.detecto.models.timeframes.interface import Timeframe


class POTTimeframe(Timeframe):
    def __init__(self) -> None:
        self.t0: float | None = None
        self.t1: float | None = None
        self.t2: float | None = None

    def set_interval(self, **params: dict[str, int | float | bool]) -> None:  # type: ignore
        total_rows = params.get("total_rows")
        t0_percentage = params.get("t0_percentage", 0.6)
        t1_percentage = params.get("t1_percentage", 0.25)
        t2_percentage = params.get("t2_percentage", 0.15)
        prod_mode = params.get("prod_mode", False)

        if prod_mode:
            t2 = 1
            t0_percentage = 0.6
            t1_percentage = 0.4
            total_rows = total_rows - t2  # type: ignore
            t0 = int(t0_percentage * total_rows)  # type: ignore
            t1 = int(t1_percentage * total_rows)  # type: ignore
            uncounted_days = t0 + t1 - total_rows  # type: ignore
        else:
            t0 = int(t0_percentage * total_rows)  # type: ignore
            t1 = int(t1_percentage * total_rows)  # type: ignore
            t2 = int(t2_percentage * total_rows)  # type: ignore
            uncounted_days = t0 + t1 + t2 - total_rows  # type: ignore

        if uncounted_days < 0:
            t1 += abs(uncounted_days)
        elif uncounted_days > 0:
            t1 -= uncounted_days

        self.t0, self.t1, self.t2 = t0, t1, t2

    def __str__(self):
        return "Peak Over Threshold Timeframe"
