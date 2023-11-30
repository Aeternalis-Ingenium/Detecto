from abc import ABCMeta, abstractmethod
from datetime import datetime


class Notification(metaclass=ABCMeta):
    @abstractmethod
    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str) -> None:
        """
        Prepares the notification with data and a custom message.

        # Parameters:
        ------------
            * data (list[dict[str, str | float | int | datetime]]): A list of dictionaries which represent all the detected anomaly data.
            * message (str): A custom message to be included in the notification.

        # Returns:
        ------------
            * None: Create JSON payload and assign it to `__payload` attribute.
        """
        pass

    @property
    @abstractmethod
    def send(self) -> None:
        """
        Sends the prepared notification. This method dispatches the notification that was set up using the `setup` method.

        # Parameters:
        ------------
            * None

        # Returns:
        ------------
            * None: This method does not return anything but is responsible for sending the notification.
        """
        pass
