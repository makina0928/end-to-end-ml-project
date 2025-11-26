from machinelearning.exception.exception import MachinelearningException
from machinelearning.logging.logger import logging

from machinelearning.entity.config_entity import DataIngestionConfig
from machinelearning.entity.artifact_entity import DataIngestionArtifact   

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL")


class DataIngestion:
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

            # Remove auto-increment IDs if present
            for col in ["id", "_id"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            # Replace placeholder missing values
            df.replace({"na": np.nan, "": np.nan}, inplace=True)

            logging.info(f"Loaded table '{table_name}' with shape {df.shape}")
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
        '''
        Orchestrates the data ingestion process:
        - Loads training, testing, and inference tables from PostgreSQL
        - Saves them as CSV files in the specified artifact paths
        - Returns a DataIngestionArtifact with the file paths
        '''
        try:
            logging.info("Starting data ingestion process...")

            # Load all 3 PostgreSQL tables
            train_df = self.load_table_as_dataframe(self.data_ingestion_config.train_table)
            test_df = self.load_table_as_dataframe(self.data_ingestion_config.test_table)
            inference_df = self.load_table_as_dataframe(self.data_ingestion_config.inference_table)

            # Save to artifacts folder
            self.save_dataframe(train_df, self.data_ingestion_config.training_file_path)
            self.save_dataframe(test_df, self.data_ingestion_config.testing_file_path)
            self.save_dataframe(inference_df, self.data_ingestion_config.inference_file_path)

            # Return DataIngestionArtifact
            ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                inference_file_path=self.data_ingestion_config.inference_file_path
            )

            logging.info(f"Data ingestion completed: {ingestion_artifact}")
            return ingestion_artifact

        except Exception as e:
            raise MachinelearningException(e, sys)
