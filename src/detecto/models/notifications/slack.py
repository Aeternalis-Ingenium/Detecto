from datetime import datetime
from http import client
from json import dumps
from urllib.parse import urlparse

from src.detecto.models.notifications.interface import Notification


class SlackNotification(Notification):
    """
    Notification class that setups message for your anomalies and sends them to Slack via webhook.

    # Attributes:
        * webhook_url (str): The URL of the Slack webhook used to send notifications.
        * __headers (dict[str, str]): The HTTP headers, by default - {"Content-Type": "application/json"}.
        * __payload (str): The payload for the notification message, by default an empty string.
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.__headers: dict[str, str] = {"Content-Type": "application/json"}
        self.__payload: str = ""
        self.__subject: str = "🤖 Detecto: Anomaly detected!"

    def __format_data(self, data: dict[str, str | float | int | datetime], index: int):
        """
        Formats a single anomaly dictionary into a string for Slack message formatting.

        # Parameters:
            * data (dict[str, str | float | int | datetime]): A dictionary containing details of an anomaly.
            * index (int): Index of the anomaly in the list, used for numbering in the message.

        # Returns:
            * str: Formatted string representing the anomaly.
        """
        date = data["date"]
        column = data["column"]
        anomaly = data["anomaly"]
        return f"{index + 1}. Date: {date} | Column: {column} | Anomaly: {anomaly}"

    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str | None):
        """
        Prepares the Slack message with given data and a custom message.

        # Parameters:
            * data (list[dict[str, str | float | int | datetime]]): A list of dictionaries which represent all the detected anomaly data.
            * message (str): A custom message to be included in the notification.

        # Returns:
            * None: Prepares the Slack message payload and assign it to `__payload` attribute.
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

        fmt_data = "\n".join(
            self.__format_data(data=anomaly_data, index=index) for index, anomaly_data in enumerate(data)
        )
        if not message:
            fmt_message = f"{self.__subject}\n" f"\n\n{fmt_data}"
        else:
            fmt_message = f"{self.__subject}\n" f"\n\n{message}\n" f"\n\n{fmt_data}"
        self.__payload = dumps({"text": fmt_message})

    @property
    def send(self):
        """
        Synchronously sends the prepared message to a Slack channel.

        # Parameters:
            * None

        # Returns:
            * None: Sends the notification to Slack and does not return anything. It prints the status of the operation.
        """
        if len(self.__payload) == 0:
            raise ValueError("Payload not set. Please call `setup()` method first.")

        parsed_url = urlparse(url=self.webhook_url)
        connection = client.HTTPSConnection(parsed_url.netloc)

        connection.request(method="POST", url=parsed_url.path, body=self.__payload, headers=self.__headers)
        response = connection.getresponse()

        if response.status == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to send notification. Status code: {response.status} - {response.reason}")

        connection.close()

    def __str__(self):
        return "Slack Notification Class"
