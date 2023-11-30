from abc import ABCMeta, abstractmethod


class Timeframe(metaclass=ABCMeta):
    @abstractmethod
    def set_interval(self, **kwargs: int | float | bool) -> None:
        """
        Sets the interval period of the dataset for Detecto model to learn from.

        # Parameters
        ------------
            * kwargs (int | float | bool): Any parameters that reflect the distribution of periods.

        # Returns
        ------------
            * None
        """
        pass
