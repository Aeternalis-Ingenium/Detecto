from typing import Literal

from src.detecto.utils._types import Notification


class FactoryNotification:
    def __init__(self, platform: Literal["email", "slack"], **kwargs: list[str] | str | int):
        self.platform = platform

        if self.platform == "email":
            self.sender_address: str = kwargs.get("sender_address")  # type: ignore
            self.password: str = kwargs.get("password")  # type: ignore
            self.recipient_addresses: list[str] = kwargs.get("recipient_addresses")  # type: ignore
            self.smtp_host: str = kwargs.get("smtp_host")  # type: ignore
            self.smtp_port: int = kwargs.get("smtp_port")  # type: ignore
        elif self.platform == "slack":
            self.webhook_url: str = kwargs.get("webhook_url")  # type: ignore

    def __call__(self) -> Notification:
        if self.platform == "email":
            from src.detecto.models.notifications.email import EmailNotification

            return EmailNotification(
                sender_address=self.sender_address,
                password=self.password,
                recipient_addresses=self.recipient_addresses,
                smtp_host=self.smtp_host,
                smtp_port=self.smtp_port,
            )
        elif self.platform == "slack":
            from src.detecto.models.notifications.slack import SlackNotification

            return SlackNotification(webhook_url=self.webhook_url)


def get_notification(platform: Literal["email", "slack"], **kwargs: list[str] | str | int) -> Notification:
    if platform == "email":
        sender_address: str = kwargs.get("sender_address")  # type: ignore
        password: str = kwargs.get("password")  # type: ignore
        recipient_addresses: list[str] = kwargs.get("recipient_addresses")  # type: ignore
        smtp_host: str = kwargs.get("smtp_host")  # type: ignore
        smtp_port: int = kwargs.get("smtp_port")  # type: ignore
        return FactoryNotification(
            platform=platform,
            sender_address=sender_address,
            password=password,
            recipient_addresses=recipient_addresses,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
        )()
    webhook_url: str = kwargs.get("webhook_url")  # type: ignore
    return FactoryNotification(platform=platform, webhook_url=webhook_url)()
