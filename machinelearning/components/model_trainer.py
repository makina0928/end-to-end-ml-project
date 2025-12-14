import os
import sys
from urllib.parse import urlparse

import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from machinelearning.exception.exception import MachinelearningException
from machinelearning.logging.logger import logging

from machinelearning.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from machinelearning.entity.config_entity import ModelTrainerConfig

from machinelearning.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)

# -----------------------------------------
# Classification Metrics
# -----------------------------------------
def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
        )
    except Exception as e:
        raise MachinelearningException(e, sys)


# -----------------------------------------
# Combined Preprocessor + Model Wrapper
# -----------------------------------------
class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise MachinelearningException(e, sys)

    def predict(self, x):
        try:
            transformed_x = self.preprocessor.transform(x)
            return self.model.predict(transformed_x)
        except Exception as e:
            raise MachinelearningException(e, sys)


# -----------------------------------------
# Model Trainer Pipeline
# -----------------------------------------
class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MachinelearningException(e, sys)

    # ------------------------------
    # MLflow Tracking
    # ------------------------------
    def track_mlflow(self, best_model, metrics: ClassificationMetricArtifact):

        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_metric("f1_score", metrics.f1_score)
            mlflow.log_metric("precision", metrics.precision_score)
            mlflow.log_metric("recall_score", metrics.recall_score)

            mlflow.sklearn.log_model(best_model, "model")

            # Register model only if MLflow backend supports it
            if tracking_scheme != "file":
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name=str(best_model),
                )

    # ------------------------------
    # Main Training Function
    # ------------------------------
    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        # -----------------------------------------
        # Identify Best Model
        # -----------------------------------------
        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # -----------------------------------------
        # Training Metrics
        # -----------------------------------------
        y_train_pred = best_model.predict(X_train)
        train_metrics = get_classification_score(y_train, y_train_pred)
        self.track_mlflow(best_model, train_metrics)

        # -----------------------------------------
        # Test Metrics
        # -----------------------------------------
        y_test_pred = best_model.predict(X_test)
        test_metrics = get_classification_score(y_test, y_test_pred)
        self.track_mlflow(best_model, test_metrics)

        # -----------------------------------------
        # Save Final Model
        # -----------------------------------------
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        os.makedirs(
            os.path.dirname(self.model_trainer_config.trained_model_file_path),
            exist_ok=True,
        )

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        # Also save raw model for model pusher
        save_object("final_model/model.pkl", best_model)

        # -----------------------------------------
        # Save test predictions for confusion matrix
        # -----------------------------------------
        os.makedirs("model_predictions", exist_ok=True)
        y_test_pred_path = "model_predictions/y_test_pred.pkl"
        save_object(y_test_pred_path, y_test_pred)

        # -----------------------------------------
        # Return Artifact (INCLUDES BEST MODEL NAME + PRED PATH)
        # -----------------------------------------
        artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metrics,
            test_metric_artifact=test_metrics,
            best_model_name=best_model_name,          
            y_test_pred_path=y_test_pred_path,         
            test_array_path=self.data_transformation_artifact.transformed_test_file_path
        )

        logging.info(f"Model trainer artifact: {artifact}")
        return artifact

    # ------------------------------
    # Starter Function
    # ------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise MachinelearningException(e, sys)