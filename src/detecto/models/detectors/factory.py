from typing import Literal

from src.detecto.utils._types import Detecto


class FactoryDetecto:
    def __init__(
        self,
        method: Literal["autoencoder", "block-maxima", "dbscan", "iso-forest", "mad", "1class-svm", "pot", "z-score"],
    ):
        self.method = method

    def __call__(self) -> Detecto:
        if self.method == "autoencoder":
            from src.detecto.models.detectors.autoencoder import AutoencoderDetecto

            return AutoencoderDetecto()
        elif self.method == "block-maxima":
            from src.detecto.models.detectors.block_maxima import BlockMaximaDetecto

            return BlockMaximaDetecto()
        elif self.method == "dbscan":
            from src.detecto.models.detectors.dbscan import DBSCANDetecto

            return DBSCANDetecto()
        elif self.method == "iso-forest":
            from src.detecto.models.detectors.isoforest import IsoForestDetecto

            return IsoForestDetecto()
        elif self.method == "mad":
            from src.detecto.models.detectors.mad import MADDetecto

            return MADDetecto()
        elif self.method == "1class-svm":
            from src.detecto.models.detectors.one_class_svm import OneClassSVMDetecto

            return OneClassSVMDetecto()
        elif self.method == "pot":
            from src.detecto.models.detectors.pot import POTDetecto

            return POTDetecto()
        from src.detecto.models.detectors.zscore import ZScoreDetecto

        return ZScoreDetecto()


def init_detecto(
    method: Literal["autoencoder", "block-maxima", "dbscan", "iso-forest", "mad", "1class-svm", "pot", "z-score"]
) -> Detecto:
    return FactoryDetecto(method=method)()
