import os
import sys
import numpy as np
import pandas as pd

""" ============================
    TRAINING PIPELINE CONSTANTS
    ============================ """

# Target column in our ML task
TARGET_COLUMN: str = "default_status"

# General pipeline metadata
PIPELINE_NAME: str = "loandefaultprediction"
ARTIFACT_DIR: str = "Artifacts"

# Output artifact filenames (these WILL be created by your pipeline)
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
INF_FILE_NAME: str = "inference.csv"


""" ============================
    POSTGRESQL DATA INGESTION CONSTANTS
    ============================ """

# PostgreSQL table names ( actual data sources)
DATA_INGESTION_TRAIN_TABLE: str = "ml_training"
DATA_INGESTION_TEST_TABLE: str = "ml_testing"
DATA_INGESTION_INFERENCE_TABLE: str = "ml_inference"

# PostgreSQL schema name
DATA_INGESTION_SCHEMA: str = "analytics"

# PostgreSQL database name
DATA_INGESTION_DB_NAME: str = "main_db"   


""" ============================
    DATA INGESTION ARTIFACT FOLDERS
    ============================ """

# Folder created under the pipeline timestamp directory
DATA_INGESTION_DIR: str = "data_ingestion"

# Subfolder where ingested CSV artifacts will be stored
DATA_INGESTION_INGESTED_DIR: str = "ingested_data"

""" ============================
    DATA VALIDATION ARTIFACT FOLDERS
    ============================ """
# Schema file path
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
# Folder created under the pipeline timestamp directory
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"