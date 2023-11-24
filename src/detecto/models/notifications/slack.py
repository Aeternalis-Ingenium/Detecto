from datetime import datetime
from http import client
from json import dumps
from urllib.parse import urlparse

from src.detecto.models.notifications.interface import Notification


class SlackNotification(Notification):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.__headers: dict[str, str] = {"Content-Type": "application/json"}
        self.__payload: str = ""

    def __format_anomaly(self, data: dict[str, str | float | int | datetime], index: int):
        """
        Formats a single anomaly dictionary into a string.
        :param anomaly: A dictionary containing details of an anomaly.
        :param index: Index of the anomaly in the list.
        """
        date = data["date"]
        column = data["column"]
        anomaly = data["anomaly"]
        return f"{index + 1}. Date: {date} | Column: {column} | Anomaly: {anomaly}"

    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str | None):
        """
        Prepares the message to be sent.
        :param data: A list of dictionaries, each containing anomaly details.
        :param message: A custom message to be included in the notification.
        """
        if type(data) != list:
            raise TypeError("Data argument must be of type list")
        else:
            for element in data:
                if type(element) != dict:
                    raise TypeError("Data argument must be of type dict")
                else:
                    for key in element.keys():
                        if key not in ["date", "column", "anomaly"]:
                            raise KeyError("Key needs to be one of these: date, column, anomaly")

        fmt_anomalies = "\n".join(
            self.__format_anomaly(data=anomaly_data, index=index) for index, anomaly_data in enumerate(data)
        )
        if not message:
            fmt_message = "🤖 Detecto: Anomaly detected!\n" f"\n\n{fmt_anomalies}"
        else:
            fmt_message = "🤖 Detecto: Anomaly detected!\n" f"\n\n{message}\n" f"\n\n{fmt_anomalies}"
        self.__payload = dumps({"text": fmt_message})

    @property
    def send(self):
        """
        Synchronously sends the prepared message to a Slack channel.
        """
        if len(self.__payload) == 0:
            raise ValueError("Payload not set. Please call `setup()` method first.")

        parsed_url = urlparse(self.webhook_url)
        connection = client.HTTPSConnection(parsed_url.netloc)

        connection.request("POST", parsed_url.path, body=self.__payload, headers=self.__headers)
        response = connection.getresponse()

        if response.status == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification. Status code: {response.status} - {response.reason}")

        connection.close()

    def __str__(self):
        return "Slack Notification Class"
