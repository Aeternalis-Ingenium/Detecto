from typing import Literal

from pandas import DataFrame

from src.detecto.models.detectors.factory import init_detecto
from src.detecto.models.notifications.factory import get_notification


class Pipeline:
    def __init__(
        self,
        dataset: DataFrame,
        temporal_feature: str,
        detecto_method: Literal[
            "autoencoder", "block-maxima", "dbscan", "iso-forest", "mad", "1class-svm", "pot", "z-score"
        ],
        notification_platform: Literal["email", "slack"],
        **kwargs: list[str] | str | int,
    ):
        self.temporal_feature = dataset[temporal_feature]
        self.dataset = dataset.drop(columns=[temporal_feature])
        self.detecto = init_detecto(method=detecto_method)
        if detecto_method == "pot":
            self.detecto.timeframe.set_interval(total_rows=self.dataset.shape[0], prod_mode=True)  # type: ignore
        if notification_platform == "email":
            self.notification = get_notification(
                platform=notification_platform,
                platform=kwargs.get("platform"),  # type: ignore
                sender_address=kwargs.get("sender_address"),  # type: ignore
                password=kwargs.get("password"),  # type: ignore
                recipient_addresses=kwargs.get("recipient_addresses"),  # type: ignore
                smtp_host=kwargs.get("smtp_host"),  # type: ignore
                smtp_port=kwargs.get("smtp_port"),  # type: ignore
            )
        elif notification_platform == "slack":
            self.notification = get_notification(platform=notification_platform, webhook_url=kwargs.get("webhook_url"))  # type: ignore

    def execute(self) -> DataFrame:
        self.detecto.compute_exceedance_threshold(dataset=self.dataset)  # type: ignore
        self.detecto.extract_exceedance(dataset=self.dataset)  # type: ignore
        self.detecto.fit(dataset=self.dataset)  # type: ignore
        self.detecto.compute_anomaly_threshold()  # type: ignore
        self.detecto.detect()  # type: ignore
        self.detecto.evaluate(method="ks", stat_distance_threshold=0.05)  # type: ignore
        #! TODO: Set a conditional based on the kstest_result "is_identical"
        #! TODO: Set a conditional to send email, slack, or both notifications
