from abc import ABCMeta, abstractmethod


class Timeframe(metaclass=ABCMeta):
    @abstractmethod
    def set_interval(self, **params: dict[str, int | float]) -> None:
        pass
