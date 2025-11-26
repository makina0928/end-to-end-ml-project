from typing import Dict, Any, List
import os
import sys
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

from machinelearning.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from machinelearning.entity.config_entity import DataValidationConfig
from machinelearning.exception.exception import MachinelearningException
from machinelearning.logging.logger import logging
from machinelearning.constants.training_pipeline import SCHEMA_FILE_PATH
from machinelearning.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """Production-grade DataValidation helper.

    Responsibilities:
      - Load schema (columns + target) from YAML
      - Validate column set (exact match)
      - Validate target column presence
      - Detect dataset drift (KS for numerical, Chi-square for categorical)
      - Produce clean YAML drift report with native Python types
      - Save valid / invalid artifacts and return DataValidationArtifact

    Notes:
      - The class expects DataValidationConfig to provide the following paths:
          * valid_train_file_path
          * valid_test_file_path
          * invalid_train_file_path
          * invalid_test_file_path
          * drift_report_file_path

    """

    def __init__(
        self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig
    ) -> None:
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema = read_yaml_file(SCHEMA_FILE_PATH) or {}
            self._expected_columns: List[str] = list(self._schema.get("columns", {}).keys())
            self._target_column: str = self._schema.get("target_column")
        except Exception as e:
            raise MachinelearningException(e, sys)

    # ------------------------- Utility IO -------------------------
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MachinelearningException(e, sys)

    # ------------------------- Schema validations -------------------------
    def validate_number_of_columns(self, dataframe: pd.DataFrame, strict: bool = True) -> bool:
        """Validate that dataframe columns match the schema.

        If strict=True then column sets must be equal (order doesn't matter).
        If strict=False then schema columns must be subset of dataframe columns.
        """
        try:
            df_cols = list(dataframe.columns)
            logging.info(f"Expected columns from schema: {self._expected_columns}")
            logging.info(f"DataFrame columns: {df_cols}")

            expected_set = set(self._expected_columns)
            df_set = set(df_cols)

            if strict:
                ok = expected_set == df_set
            else:
                ok = expected_set.issubset(df_set)

            if not ok:
                logging.error(f"Column mismatch. expected={sorted(expected_set)} found={sorted(df_set)}")

            return ok
        except Exception as e:
            raise MachinelearningException(e, sys)

    def validate_target_column(self, df: pd.DataFrame) -> bool:
        """Ensure target column is present in DataFrame and schema."""
        try:
            if not self._target_column:
                logging.error("target_column missing in schema.yaml")
                return False

            if self._target_column not in df.columns:
                logging.error(f"Target column '{self._target_column}' missing in DataFrame")
                return False

            logging.info(f"✔ Target column '{self._target_column}' is present.")
            return True
        except Exception as e:
            raise MachinelearningException(e, sys)

    # ------------------------- Drift detection -------------------------
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """Detect drift between base (train) and current (test/inference) datasets.

        Returns True if **no drift** detected for any column, False if any column drifts.
        Also writes a human-readable YAML report to data_validation_config.drift_report_file_path.
        """
        try:
            status = True
            report: Dict[str, Any] = {}

            # operate only on common columns to avoid KeyError
            common_columns = [c for c in self._expected_columns if c in base_df.columns and c in current_df.columns]
            if not common_columns:
                # fallback to any intersection
                common_columns = list(set(base_df.columns) & set(current_df.columns))

            logging.info(f"Drift check columns: {common_columns}")

            for column in common_columns:
                base_col = base_df[column]
                curr_col = current_df[column]

                # Numerical: KS test
                if pd.api.types.is_numeric_dtype(base_col):
                    try:
                        ks_result = ks_2samp(base_col.dropna(), curr_col.dropna())
                        p_val = float(ks_result.pvalue)
                        drift = bool(p_val < threshold)

                        report[column] = {"type": "numerical", "p_value": p_val, "drift_status": drift}

                        if drift:
                            logging.warning(f"[DRIFT DETECTED] {column} (p={p_val})")
                            status = False
                        else:
                            logging.info(f"[NO DRIFT] {column} (p={p_val})")
                    except Exception as e:
                        logging.exception(f"Failed KS test for column {column}: {e}")
                        # record an entry to the report and continue
                        report[column] = {"type": "numerical", "error": str(e)}
                        status = False

                # Categorical/object: Chi-square on counts
                elif pd.api.types.is_object_dtype(base_col) or pd.api.types.is_categorical_dtype(base_col):
                    try:
                        base_counts = base_col.astype(str).value_counts()
                        curr_counts = curr_col.astype(str).value_counts()

                        all_categories = list(set(base_counts.index) | set(curr_counts.index))

                        contingency = [
                            [int(base_counts.get(cat, 0)) for cat in all_categories],
                            [int(curr_counts.get(cat, 0)) for cat in all_categories],
                        ]

                        # chi2 test requires at least 2 categories and non-empty counts
                        if len(all_categories) < 2 or sum(contingency[0]) == 0 or sum(contingency[1]) == 0:
                            logging.info(f"Skipping chi2 for {column} due to insufficient categories or counts")
                            report[column] = {"type": "categorical", "p_value": None, "drift_status": False}
                            continue

                        chi2, p_val, dof, expected = chi2_contingency(contingency)
                        p_val = float(p_val)
                        drift = bool(p_val < threshold)

                        report[column] = {"type": "categorical", "p_value": p_val, "drift_status": drift}

                        if drift:
                            logging.warning(f"[DRIFT DETECTED] {column} (p={p_val})")
                            status = False
                        else:
                            logging.info(f"[NO DRIFT] {column} (p={p_val})")

                    except Exception as e:
                        logging.exception(f"Failed chi2 test for column {column}: {e}")
                        report[column] = {"type": "categorical", "error": str(e)}
                        status = False

                else:
                    logging.info(f"Skipping column {column} (unsupported dtype: {base_col.dtype})")
                    report[column] = {"type": "unsupported", "dtype": str(base_col.dtype)}

            # ensure report contains native python types (no numpy scalars)
            for col, stats in report.items():
                if "p_value" in stats and stats["p_value"] is not None:
                    stats["p_value"] = float(stats["p_value"])  # safe conversion
                if "drift_status" in stats:
                    stats["drift_status"] = bool(stats["drift_status"])  # ensure python bool

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise MachinelearningException(e, sys)

    # ------------------------- Orchestration -------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # column structure
            train_columns_ok = self.validate_number_of_columns(train_df, strict=True)
            test_columns_ok = self.validate_number_of_columns(test_df, strict=True)

            if not (train_columns_ok and test_columns_ok):
                logging.error("Column structure mismatch. Failing validation.")
                return self._fail_due_to_column_mismatch(train_df, test_df)

            # target presence
            target_ok_train = self.validate_target_column(train_df)
            target_ok_test = self.validate_target_column(test_df)
            if not (target_ok_train and target_ok_test):
                logging.error("Target column missing. Failing validation.")
                return self._fail_due_to_missing_target(train_df, test_df)

            # drift detection
            drift_ok = self.detect_dataset_drift(train_df, test_df)

            # Save validated datasets to "valid" paths
            valid_dir = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(valid_dir, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            return DataValidationArtifact(
                validation_status=drift_ok,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

        except Exception as e:
            raise MachinelearningException(e, sys)

    # ------------------------- Failure helpers -------------------------
    def _fail_due_to_missing_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> DataValidationArtifact:
        try:
            invalid_dir = os.path.dirname(self.data_validation_config.invalid_train_file_path)
            os.makedirs(invalid_dir, exist_ok=True)

            train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)

            report = {"error": "target_column_missing"}
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report, replace=True)

            return DataValidationArtifact(
                validation_status=False,
                valid_train_file_path=None,
                valid_test_file_path=None,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise MachinelearningException(e, sys)

    def _fail_due_to_column_mismatch(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> DataValidationArtifact:
        try:
            invalid_dir = os.path.dirname(self.data_validation_config.invalid_train_file_path)
            os.makedirs(invalid_dir, exist_ok=True)

            train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)

            report = {"error": "column_mismatch"}
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report, replace=True)

            return DataValidationArtifact(
                validation_status=False,
                valid_train_file_path=None,
                valid_test_file_path=None,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise MachinelearningException(e, sys)
