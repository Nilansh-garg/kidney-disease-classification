from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import os 
import urllib.request as request
import zipfile
from pathlib import Path
import tensorflow as tf
from cnnClassifier import logger


class prepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    def get_base_model(self, model_type: str):
        """Downloads the raw base model from keras applications """
        
        if model_type == 'densenet121':
            model_fn = tf.keras.applications.DenseNet121
        elif model_type == 'resnet50':
            model_fn = tf.keras.applications.ResNet50
        elif model_type == 'efficientnetb0':
            model_fn = tf.keras.applications.EfficientNetB0
            
        self.model = model_fn(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            pooling=self.config.params_pooling,
        )
        
        self.save_model(model = self.model, path = self.config.base_model_path)
        
    @staticmethod 
    def save_model(model: tf.keras.Model, path: Path):
        """ Saves the keras model to the path """
        str_path = str(path)
        model.save(str_path)
        
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till,learning_rate):
        """ Creates the full model with custom head """
        
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
                
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
                
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(model.output)
        
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        
        full_model.summary()
        print(f"Model Output Shape: {full_model.output_shape}")
        return full_model
    
    def update_base_model(self):
        """ Updates the base model with custom head """
        
        full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        self.save_model(model=full_model, path=self.config.updated_base_model_path)