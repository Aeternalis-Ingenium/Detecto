from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP

from src.detecto.models.notifications.interface import Notification


class EmailNotification(Notification):
    """
    Notification class that setups the message for your anomalies and sends them via your email address.

    # Attributes:
    ------------
        * sender_address (str): The email address from which the email will be sent.
        * password (str): The password or app-specific password for authenticating the email account.
        * recipient_addresses (list[str]): The list of your recipent's email addresses.
        * smtp_host (str): The SMTP host address for the email provider: smtp.YOUR_EMAIL_PROVIDER.com.
        * smtp_port (int): The SMTP server port for the email provider, default 587.
        * __payload (str): The payload for the notification message, by default an empty string.
        * __subject (str): The sibject of your email.
    """

    def __init__(
        self,
        sender_address: str,
        password: str,
        recipient_addresses: list[str],
        smtp_host: str,
        smtp_port: int = 587,
    ) -> None:
        self.sender_address = sender_address
        self.password = password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.recipient_addresses = recipient_addresses
        self.__subject = "🤖 Detecto: Anomaly detected!"
        self.__payload: str = ""

    def __format_data(self, data: dict[str, str | float | int | datetime], index: int):
        """
        Formats a single anomaly dictionary into a string for Slack message formatting.

        # Parameters:
        ------------
            * data (dict[str, str | float | int | datetime]): A dictionary containing details of an anomaly.
            * index (int): Index of the anomaly in the list, used for numbering in the message.

        # Returns:
        ------------
            * str: Formatted string representing the anomaly.
        """
        date = data["date"]
        column = data["column"]
        anomaly = data["anomaly"]
        return f"{index + 1}. Date: {date} | Column: {column} | Anomaly: {anomaly}"

    def setup(self, data: list[dict[str, str | float | int | datetime]], message: str):
        """
        Prepares the email message with given data and a custom message.

        # Parameters:
        ------------
            * data (list[dict[str, str | float | int | datetime]]): Data to be included in the email.
            * message (str): A custom message to be included in the email.

        # Returns:
        ------------
            * None: This method prepares the email content.
        """
        fmt_data = "\n".join(
            self.__format_data(data=anomaly_data, index=index) for index, anomaly_data in enumerate(data)
        )
        fmt_message = f"{message}\n\n{fmt_data}"
        self.__payload = fmt_message
        pass

    @property
    def send(self):
        """
        Synchronously sends the prepared email to the specified addresses.

        # Parameters:
        ------------
            * None

        # Returns:
        ------------
            * None: Sends notification via sender address and prints the status of the operation.
        """
        if len(self.__payload) == 0:
            raise ValueError("Payload not set. Please call `setup()` method first.")

        msg = MIMEMultipart()
        msg["From"] = self.sender_address
        msg["To"] = ", ".join(self.recipient_addresses)
        msg["Subject"] = self.__subject
        msg.attach(MIMEText(self.__payload, "plain"))

        try:
            server = SMTP(host=self.smtp_host, port=self.smtp_port)
            server.starttls()
            server.login(user=self.sender_address, password=self.password)
            server.send_message(msg=msg, from_addr=self.sender_address, to_addrs=self.recipient_addresses)
            server.quit()
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email. Error: {e}")

    def __str__(self):
        return "Email Notification"
