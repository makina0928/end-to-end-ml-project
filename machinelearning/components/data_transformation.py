import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline

from machinelearning.constants.training_pipeline import TARGET_COLUMN
from machinelearning.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from machinelearning.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from machinelearning.entity.config_entity import DataTransformationConfig
from machinelearning.exception.exception import MachinelearningException 
from machinelearning.logging.logger import logging
from machinelearning.utils.main_utils.utils import save_numpy_array_data, save_object
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MachinelearningException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MachinelearningException(e, sys)
        
    # def get_data_transformer_object(self) -> Pipeline:
    #     """
    #     Returns a pipeline with KNNImputer for numeric columns.
    #     """
    #     logging.info("Entered get_data_transformer_object method")
    #     try:
    #         imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
    #         logging.info(f"Initialised KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
    #         processor = Pipeline([("imputer", imputer)])
    #         return processor

    #     except Exception as e:
    #         raise MachinelearningException(e, sys)

    def get_data_transformer_object(self, numeric_cols, categorical_cols):
        try:
            logging.info("Building preprocessing pipeline")

            # Numeric pipeline
            numeric_pipeline = Pipeline(
                steps=[
                    ("num_imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("cat_imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise MachinelearningException(e, sys)
     
    # def initiate_data_transformation(self) -> DataTransformationArtifact:
    #     logging.info("Entered initiate_data_transformation method of DataTransformation class")
    #     try:
    #         logging.info("Starting data transformation")
    #         train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
    #         test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

    #         # split target and features
    #         input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
    #         target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

    #         input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
    #         target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

    #         # ================================
    #         #SEPARATE NUMERIC & CATEGORICAL
    #         # ================================
    #         numeric_cols = input_feature_train_df.select_dtypes(include=[np.number]).columns
    #         categorical_cols = input_feature_train_df.select_dtypes(exclude=[np.number]).columns

    #         logging.info(f"Numeric columns: {list(numeric_cols)}")
    #         logging.info(f"Categorical columns: {list(categorical_cols)}")

    #         # ================================
    #         # KNN IMPUTE NUMERIC COLUMNS
    #         # ================================
    #         preprocessor = self.get_data_transformer_object()
    #         numeric_train = preprocessor.fit_transform(input_feature_train_df[numeric_cols])
    #         numeric_test = preprocessor.transform(input_feature_test_df[numeric_cols])

    #         numeric_train_df = pd.DataFrame(numeric_train, columns=numeric_cols)
    #         numeric_test_df = pd.DataFrame(numeric_test, columns=numeric_cols)

    #         # ================================
    #         # IMPUTE CATEGORICAL COLUMNS
    #         # ================================
    #         cat_imputer = SimpleImputer(strategy="most_frequent")

    #         categorical_train = cat_imputer.fit_transform(input_feature_train_df[categorical_cols])
    #         categorical_test = cat_imputer.transform(input_feature_test_df[categorical_cols])

    #         categorical_train_df = pd.DataFrame(categorical_train, columns=categorical_cols)
    #         categorical_test_df = pd.DataFrame(categorical_test, columns=categorical_cols)

    #         # ================================
    #         # COMBINE NUMERIC + CATEGORICAL BACK
    #         # ================================
    #         final_train_df = pd.concat([numeric_train_df, categorical_train_df], axis=1)
    #         final_test_df = pd.concat([numeric_test_df, categorical_test_df], axis=1)

    #         # ================================
    #         #APPEND TARGET COLUMN
    #         # ================================
    #         train_arr = np.c_[final_train_df.values, np.array(target_feature_train_df)]
    #         test_arr = np.c_[final_test_df.values, np.array(target_feature_test_df)]

    #         # ================================
    #         #SAVE TRANSFORMED ARTIFACTS
    #         # ================================
    #         save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
    #         save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

    #         # Save both imputers
    #         transformer_bundle = {
    #             "numeric_imputer": preprocessor,
    #             "categorical_imputer": cat_imputer,
    #             "numeric_cols": list(numeric_cols),
    #             "categorical_cols": list(categorical_cols)
    #         }

    #         save_object(self.data_transformation_config.transformed_object_file_path, transformer_bundle)
    #         save_object("final_model/preprocessor.pkl", transformer_bundle)

    #         # ================================
    #         # RETURN ARTIFACT
    #         # ================================
    #         data_transformation_artifact = DataTransformationArtifact(
    #             transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
    #             transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
    #             transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
    #         )
    #         return data_transformation_artifact
        
    #     except Exception as e:
    #         raise MachinelearningException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Separate target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].replace(-1, 0)

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].replace(-1, 0)

            # Identify numeric & categorical columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

            logging.info(f"Numeric columns: {list(numeric_cols)}")
            logging.info(f"Categorical columns: {list(categorical_cols)}")

            # Build preprocessing pipeline
            preprocessor = self.get_data_transformer_object(numeric_cols, categorical_cols)

            # Fit-transform training, transform testing
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine features + target
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save the preprocessor
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_object("final_model/preprocessor.pkl", preprocessor)

            # Build artifact
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise MachinelearningException(e, sys)
