from unittest import TestCase

from src.detecto import init_detecto
from src.detecto.models.detectors.autoencoder import AutoencoderDetecto
from src.detecto.models.detectors.block_maxima import BlockMaximaDetecto
from src.detecto.models.detectors.dbscan import DBSCANDetecto
from src.detecto.models.detectors.interface import Detecto
from src.detecto.models.detectors.isoforest import IsoForestDetecto
from src.detecto.models.detectors.mad import MADDetecto
from src.detecto.models.detectors.one_class_svm import OneClassSVMDetecto
from src.detecto.models.detectors.pot import POTDetecto
from src.detecto.models.detectors.zscore import ZScoreDetecto


class TestFactoryDetecto(TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_construct_autoencoder_detecto_from_factory_design_pattern(self):
        autoencoder_detecto = init_detecto(method="autoencoder")

        self.assertTrue(expr=issubclass(type(autoencoder_detecto), Detecto))
        self.assertIsInstance(obj=autoencoder_detecto, cls=AutoencoderDetecto)
        self.assertEqual(first=str(autoencoder_detecto), second="Autoencoder Anomaly Detector")

    def test_construct_blockmaxima_detecto_from_factory_design_pattern(self):
        blockmaxima_detecto = init_detecto(method="block-maxima")

        self.assertTrue(expr=issubclass(type(blockmaxima_detecto), Detecto))
        self.assertIsInstance(obj=blockmaxima_detecto, cls=BlockMaximaDetecto)
        self.assertEqual(first=str(blockmaxima_detecto), second="Block-Maxima Anomaly Detector")

    def test_construct_dbscan_detecto_from_factory_design_pattern(self):
        dbscan_detecto = init_detecto(method="dbscan")

        self.assertTrue(expr=issubclass(type(dbscan_detecto), Detecto))
        self.assertIsInstance(obj=dbscan_detecto, cls=DBSCANDetecto)
        self.assertEqual(
            first=str(dbscan_detecto),
            second="Density-Based Spatial Clustering of Applications with Noise Anomaly Detector",
        )

    def test_construct_isoforest_detecto_from_factory_design_pattern(self):
        iso_forest_detecto = init_detecto(method="iso-forest")

        self.assertTrue(expr=issubclass(type(iso_forest_detecto), Detecto))
        self.assertIsInstance(obj=iso_forest_detecto, cls=IsoForestDetecto)
        self.assertEqual(first=str(iso_forest_detecto), second="Isolation Forest Anomaly Detector")

    def test_construct_mad_detecto_from_factory_design_pattern(self):
        mad_detecto = init_detecto(method="mad")

        self.assertTrue(expr=issubclass(type(mad_detecto), Detecto))
        self.assertIsInstance(obj=mad_detecto, cls=MADDetecto)
        self.assertEqual(first=str(mad_detecto), second="Mean Absolute Deviation Anomaly Detector")

    def test_construct_1_class_svm_detecto_from_factory_design_pattern(self):
        one_class_svm_detecto = init_detecto(method="1class-svm")

        self.assertTrue(expr=issubclass(type(one_class_svm_detecto), Detecto))
        self.assertIsInstance(obj=one_class_svm_detecto, cls=OneClassSVMDetecto)
        self.assertEqual(first=str(one_class_svm_detecto), second="One Class Support Vector Machine Anomaly Detector")

    def test_construct_pot_detecto_from_factory_design_pattern(self):
        pot_detecto = init_detecto(method="pot")

        self.assertTrue(expr=issubclass(type(pot_detecto), Detecto))
        self.assertIsInstance(obj=pot_detecto, cls=POTDetecto)
        self.assertEqual(first=str(pot_detecto), second="Peak Over Threshold Anomaly Detector")

    def test_construct_zscore_detecto_from_factory_design_pattern(self):
        z_score_detecto = init_detecto(method="z-score")  # type: ignore

        self.assertTrue(expr=issubclass(type(z_score_detecto), Detecto))
        self.assertIsInstance(obj=z_score_detecto, cls=ZScoreDetecto)
        self.assertEqual(first=str(z_score_detecto), second="Z-Score Anomaly Detector")

    def tearDown(self) -> None:
        return super().tearDown()
