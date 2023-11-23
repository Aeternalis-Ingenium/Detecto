from abc import ABCMeta, abstractmethod
from datetime import datetime


class Notification(metaclass=ABCMeta):
    @abstractmethod
    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str) -> None:
        """_summary_

        Args:
            data (_type_): _description_
        """

    @property
    @abstractmethod
    def send(self) -> None:
        """_summary_

        Args:
            data (_type_): _description_
        """
