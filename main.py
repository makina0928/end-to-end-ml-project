import os
import sys
from machinelearning.logging.logger import logging
from machinelearning.exception.exception import MachinelearningException
from machinelearning.entity.config_entity import TrainingPipelineConfig
from machinelearning.components.data_ingestion import DataIngestion
from machinelearning.entity.config_entity import DataIngestionConfig, DataValidationConfig
from machinelearning.components.data_validation import DataValidation

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)

    except Exception as e:
        raise MachinelearningException(e, sys)