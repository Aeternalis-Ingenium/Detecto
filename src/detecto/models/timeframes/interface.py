from abc import ABCMeta, abstractmethod


class Timeframe(metaclass=ABCMeta):
    @abstractmethod
    def set_interval(self, **kwargs: int | float | bool) -> None:
        pass
