from machinelearning.exception.exception import MachinelearningException
from machinelearning.logging.logger import logging

from machinelearning.entity.config_entity import DataIngestionConfig
from machinelearning.entity.artifact_entity import DataIngestionArtifact   

import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL")


class DataIngestion:

    COLUMNS_TO_DROP = [
        "fact_customer_id",
        "dim_customer_id",
        "fact_branch_id",
        "dim_branch_id",
        "loan_id",
        "par_0",
        "par_30",
        "par_60",
        "par_90",
        "disbursement_month",
        "disbursement_year",
        "income_bracket",
        "disbursement_date"
    ]

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MachinelearningException(e, sys)

    # ----------------------------
    # Load Any PostgreSQL table
    # ----------------------------
    def load_table_as_dataframe(self, table_name: str) -> pd.DataFrame:
        try:
            logging.info(f"Loading table '{table_name}' from PostgreSQL")

            engine = create_engine(POSTGRES_DB_URL)

            schema = self.data_ingestion_config.schema
            table = table_name

            query = f'SELECT * FROM "{schema}"."{table}";'
            df = pd.read_sql(query, engine)

            # Remove auto-generated IDs
            for col in ["id", "_id"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            # Replace placeholder missing values
            df.replace({"na": np.nan, "": np.nan}, inplace=True)

            # Drop unwanted columns (ignore missing)
            df.drop(columns=self.COLUMNS_TO_DROP, errors="ignore", inplace=True)

            logging.info(
                f"Loaded table '{table_name}' | Final shape after cleaning: {df.shape}"
            )
            return df

        except Exception as e:
            raise MachinelearningException(e, sys)

    # ----------------------------
    # Save DataFrame to CSV
    # ----------------------------
    def save_dataframe(self, df: pd.DataFrame, file_path: str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False, header=True)
            logging.info(f"Saved DataFrame to: {file_path}")

        except Exception as e:
            raise MachinelearningException(e, sys)

    # ----------------------------
    # Orchestrate full ingestion
    # ----------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process...")

            # Load PostgreSQL tables
            train_df = self.load_table_as_dataframe(self.data_ingestion_config.train_table)
            test_df = self.load_table_as_dataframe(self.data_ingestion_config.test_table)
            inference_df = self.load_table_as_dataframe(self.data_ingestion_config.inference_table)

            # Save cleaned datasets
            self.save_dataframe(train_df, self.data_ingestion_config.training_file_path)
            self.save_dataframe(test_df, self.data_ingestion_config.testing_file_path)
            self.save_dataframe(inference_df, self.data_ingestion_config.inference_file_path)

            ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                inference_file_path=self.data_ingestion_config.inference_file_path
            )

            logging.info(f"Data ingestion completed successfully: {ingestion_artifact}")
            return ingestion_artifact

        except Exception as e:
            raise MachinelearningException(e, sys)
