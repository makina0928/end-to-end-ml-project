from datetime import datetime
import os
from machinelearning.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")):
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(
            self.artifact_name, timestamp
        )
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Artifact folders
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR
        )

        # Output CSV artifact paths
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        self.inference_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.INF_FILE_NAME
        )

        # PostgreSQL tables
        self.train_table = training_pipeline.DATA_INGESTION_TRAIN_TABLE
        self.test_table = training_pipeline.DATA_INGESTION_TEST_TABLE
        self.inference_table = training_pipeline.DATA_INGESTION_INFERENCE_TABLE

        # PostgreSQL database name and schema
        self.database_name = training_pipeline.DATA_INGESTION_DB_NAME
        self.schema = training_pipeline.DATA_INGESTION_SCHEMA

