import time
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Optional: helps debug data issues
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
model = tf.keras.models.load_model('artifacts/training/model.h5')
import mlflow
import mlflow.keras


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def _valid_generator(self):
        
        datagenrator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )
        dataflow_kwargs = dict(
            target_size=(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="binary"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenrator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        
        # ADD THIS — recompile with full metrics
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        
    def save_score(self):
        precision = self.score[3]
        recall = self.score[4]
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # 1e-7 avoids division by zero
        
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "auc": self.score[2],
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        save_json(path=Path("scores.json"), data=scores)
    def log_into_mlflow(self):
        precision = self.score[3]
        recall = self.score[4]
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("kidney-disease-classification")
        
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
                "auc": self.score[2],
                "precision": self.score[3],
                "recall": self.score[4],
                "f1_score": f1          # ADD THIS
            })
            if self.model:
                mlflow.keras.log_model(self.model, "model", registered_model_name="KidneyDiseaseModel")
        
        # Log the actual Model (Optional but highly recommended)
        # This allows you to deploy the model directly from DagsHub/MLflow later
        if self.model:
            mlflow.keras.log_model(self.model, "model", registered_model_name="KidneyDiseaseModel")
        