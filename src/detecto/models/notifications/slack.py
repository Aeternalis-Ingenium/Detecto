from datetime import datetime

from detecto.models.notifications.interface import Notification


class SlackNotification(Notification):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.__headers = {"Content-Type": "application/json"}
        self.__payload = None

    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str):
        pass

    @property
    def send(self):
        pass
