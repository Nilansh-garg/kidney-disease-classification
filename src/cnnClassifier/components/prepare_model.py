import os
import urllib.request as request
from zipfile import ZipFile
from cnnClassifier import logger
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        logger.info("Downloading base model")
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape = self.config.param_image_size,
            include_top = self.config.param_include_top,
            weights = self.config.param_weights
        )

        logger.info("Saving base model")
        base_model = self.model
        
        self.save_model(path=self.config.base_model_path, model=base_model)

        
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable= False
        elif(freeze_till is not None) and (freeze_till> 0):
            for layer in model.layers[:-freeze_till]:
                 model.trainable = False
                 
        flatten_in = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dense(256, activation="relu")(flatten_in)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
        
        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = output
        )
        
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics = ["accuracy"]
        )
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
        model = self.model,
        classes= self.config.params_classes,
        freeze_all = False,
        freeze_till = 4,
        learning_rate = self.config.param_learning_rate
        )
        
        self.save_model(path= self.config.updated_base_model_path, model = self.full_model)
        
    @staticmethod
    
    def save_model(path:Path, model: tf.keras.Model):
        model.save(path)
    
    