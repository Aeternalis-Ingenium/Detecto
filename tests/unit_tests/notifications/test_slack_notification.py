from json import dumps
from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.detecto.models.notifications.slack import SlackNotification


class TestSlackNotification(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.slack_notification = SlackNotification(webhook_url="https://hooks.slack.com/services/TEST/TOKEN/WEBHOOK")
        self.test_message = "Test notification message"
        self.test_data = [
            {"date": "2023-10-23T00:00:00.000Z", "column": "col_1", "anomaly": 2003.214},
            {"date": "2023-10-23T00:00:00.000Z", "column": "col_2", "anomaly": 1055.67},
        ]

    def test_setup(self):
        expected_payload = dumps(
            {
                "text": (
                    "🤖 Detecto: Anomaly detected!\n"
                    f"{self.test_message}\n"
                    "1. Date: 2023-10-23T00:00:00.000Z | Column: col_1 | Anomaly: 2003.214"
                    "\n2. Date: 2023-10-23T00:00:00.000Z | Column: col_2 | Anomaly: 1055.67"
                )
            }
        )

        self.slack_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore

        self.assertIsNotNone(self.slack_notification._SlackNotification__payload)
        self.assertEqual(first=self.slack_notification._SlackNotification__payload, second=expected_payload)

    @patch("src.detecto.models.notifications.slack.client.HTTPSConnection")
    def test_send_method(self, mock_https_connection):
        mock_connection = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_connection.getresponse.return_value = mock_response
        mock_https_connection.return_value = mock_connection

        self.slack_notification.setup(data=self.test_data, message=self.test_message)  # type: ignore
        self.slack_notification.send()

        mock_connection.request.assert_called_once()
        mock_connection.close.assert_called_once()

    def test_send_without_setup(self):
        with self.assertRaises(ValueError):
            self.slack_notification.send()

    def tearDown(self) -> None:
        return super().tearDown()
