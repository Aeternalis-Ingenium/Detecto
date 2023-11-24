from datetime import datetime

from detecto.models.notifications.interface import Notification


class EmailNotification(Notification):
    """
    Notification class that setups the message for your anomalies and sends them via your email address.

    # Attributes:
        * sender_address (str): The email address from which the email will be sent.
        * password (str): The password or app-specific password for authenticating the email account.
        * recipient_addresses (list[str]): The list of your recipent's email addresses.
        * smtp_server (str): The SMTP server address for the email provider: smtp.YOUR_EMAIL_PROVIDER.com.
        * smtp_port (int): The SMTP server port for the email provider, default 587.
        * __payload (str): The payload for the notification message, by default an empty string.
        * __subject (str): The sibject of your email.
    """

    def __init__(
        self,
        sender_address: str,
        password: str,
        recipient_addresses: list[str],
        smtp_server: str,
        smtp_port: int = 587,
    ) -> None:
        self.sender_address = sender_address
        self.password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.recipient_addresses = recipient_addresses
        self.__subject = "🤖 Detecto: Anomaly detected!"
        self.__payload: str = ""

    def __format_anomaly(self, anomaly: dict[str, str | float | int | datetime], index: int) -> str:  # type: ignore
        """
        Formats a single anomaly dictionary into a string for email message formatting.

        Parameters:
            anomaly (dict[str, str | float | int | datetime]): A dictionary containing details of an anomaly.
            index (int): Index of the anomaly in the list, used for numbering in the message.

        Returns:
            str: Formatted string representing the anomaly.
        """
        pass

    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str):
        """
        Prepares the email message with given data and a custom message.

        Parameters:
            data (list[dict[str, str | float | int | datetime]]): Data to be included in the email.
            message (str): A custom message to be included in the email.

        Returns:
            None: This method prepares the email content.
        """
        pass

    @property
    def send(self):
        """
        Synchronously sends the prepared email to the specified addresses.

        Parameters:
            * None

        Returns:
            None: Sends notification via sender address and prints the status of the operation.
        """
        pass
