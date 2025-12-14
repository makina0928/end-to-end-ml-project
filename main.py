import os
import sys
from machinelearning.logging.logger import logging
from machinelearning.exception.exception import MachinelearningException
from machinelearning.entity.config_entity import TrainingPipelineConfig
from machinelearning.components.data_ingestion import DataIngestion
from machinelearning.components.data_validation import DataValidation
from machinelearning.components.data_transformation import DataTransformation
from machinelearning.components.model_trainer import ModelTrainer
from machinelearning.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

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
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiate the data Transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info("data Transformation Completed")
        print(data_transformation_artifact)
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config,data_transformation_artifact)
        logging.info("Initiate the model trainer")
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info("model trainer Completed")
        print(model_trainer_artifact)
    except Exception as e:
        raise MachinelearningException(e, sys)