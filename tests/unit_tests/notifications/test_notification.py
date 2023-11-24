from unittest import TestCase

from tests.conftest import AbstractNotificationTestModel


class TestNotificationInterfaceClass(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.notification = AbstractNotificationTestModel()

    def test_setup_method(self):
        self.assertIsNone(
            obj=self.notification.setup(
                data=[{"date": "2023-10-23", "column": "col_1", "anomaly": 2023.23}], message="Testing anomaly notification"  # type: ignore
            )
        )

    def test_set_interval_method_prod_mode(self):
        self.assertIsNone(obj=self.notification.send)  # type: ignore

    def tearDown(self) -> None:
        return super().tearDown()
