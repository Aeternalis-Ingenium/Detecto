from src.detecto.models.detectors.autoencoder import AutoencoderDetecto
from src.detecto.models.detectors.block_maxima import BlockMaximaDetecto
from src.detecto.models.detectors.dbscan import DBSCANDetecto
from src.detecto.models.detectors.isoforest import IsoForestDetecto
from src.detecto.models.detectors.mad import MADDetecto
from src.detecto.models.detectors.one_class_svm import OneClassSVMDetecto
from src.detecto.models.detectors.pot import POTDetecto
from src.detecto.models.detectors.zscore import ZScoreDetecto
from src.detecto.models.notifications.email import EmailNotification
from src.detecto.models.notifications.slack import SlackNotification

Detecto = (
    AutoencoderDetecto
    | BlockMaximaDetecto
    | DBSCANDetecto
    | IsoForestDetecto
    | MADDetecto
    | OneClassSVMDetecto
    | POTDetecto
    | ZScoreDetecto
)

Notification = EmailNotification | SlackNotification
