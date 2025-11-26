import os
import sys
from machinelearning.logging.logger import logging
from machinelearning.exception.exception import MachinelearningException
from machinelearning.entity.config_entity import TrainingPipelineConfig
from machinelearning.components.data_ingestion import DataIngestion
from machinelearning.entity.config_entity import DataIngestionConfig
from machinelearning.entity.artifact_entity import DataIngestionArtifact

def main():
    try:
        # Initialize Data Ingestion Configuration
        logging.info("Initializing Data Ingestion Configuration...")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        # Create Data Ingestion Component
        logging.info("Creating Data Ingestion Component...")
        data_ingestion = DataIngestion(data_ingestion_config)

        # Start Data Ingestion Process
        logging.info("Starting Data Ingestion Process...")
        ingestion_artifact: DataIngestionArtifact = data_ingestion.initiate_data_ingestion()

        logging.info(f"Data Ingestion Artifact Completed: {ingestion_artifact}")

    except Exception as e:
        raise MachinelearningException(e, sys)
    
if __name__ == "__main__":
    main()