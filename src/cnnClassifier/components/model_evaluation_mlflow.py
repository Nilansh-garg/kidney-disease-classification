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
            validation_split=0.30
        )
        dataflow_kwargs = dict(
            target_size=(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
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
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)
        
    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1]
        }
        save_json(
            path=Path("scores.json"),
            data=scores
        )
    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("kidney-disease-classification")
        
        with mlflow.start_run(run_name="evaluation"):
            # Log Hyperparameters
            mlflow.log_params(self.config.all_params)
            
            # Log Metrics
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })
            
            # Log the actual Model (Optional but highly recommended)
            # This allows you to deploy the model directly from DagsHub/MLflow later
            if self.model:
                mlflow.keras.log_model(self.model, "model", registered_model_name="KidneyDiseaseModel")
            