from abc import ABCMeta, abstractmethod


class Notification(metaclass=ABCMeta):
    @abstractmethod
    def setup(self, data) -> None:
        """_summary_

        Args:
            data (_type_): _description_
        """

    @abstractmethod
    async def send(self) -> None:
        """_summary_

        Args:
            data (_type_): _description_
        """
