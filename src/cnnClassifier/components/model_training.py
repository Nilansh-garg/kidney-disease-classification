import time
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Optional: helps debug data issues
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    def train_valid_generator(self):
        datagen_kwargs = dict(rescale=1./255, validation_split=0.20)
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            class_mode = "categorical",
        )
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
        self.valid_generator = valid_datagen.flow_from_directory(
            directory = self.config.training_data,
            subset = "validation",
            shuffle = True,
            **dataflow_kwargs
        )
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                shear_range=0.2,
                zoom_range=0.2,
                **datagen_kwargs
            )
        else:
            train_datagen = valid_datagen
            
        self.train_generator = train_datagen.flow_from_directory(
            directory = self.config.training_data,
            subset = "training",
            shuffle = True,
            **dataflow_kwargs
        )
        
    @staticmethod
    def save_model(model:tf.keras.Model, path:Path)->None:
        model.save(path)
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
            
        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epochs,
            validation_data = self.valid_generator,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
        )
            
        self.save_model(self.model, self.config.trained_model_path)