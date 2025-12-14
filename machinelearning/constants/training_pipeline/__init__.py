import os
import sys
import numpy as np
import pandas as pd

""" ============================
    TRAINING PIPELINE CONSTANTS
    ============================ """

# Target column in our ML task
TARGET_COLUMN: str = "target_default"

# General pipeline metadata
PIPELINE_NAME: str = "loandefaultprediction"
ARTIFACT_DIR: str = "Artifacts"

# Output artifact filenames (these WILL be created by your pipeline)
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
INF_FILE_NAME: str = "inference.csv"

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"


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

"""======================================================================
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
======================================================================"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

"""============================================================
Model Trainer ralated constant start with MODE TRAINER VAR NAME
============================================================"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05