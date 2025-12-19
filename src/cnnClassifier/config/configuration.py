from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath= PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzipped_data_dir=config.unzipped_data_dir
        )
        
        return data_ingestion_config
    
    def _create_config_for(self, model_name: str) -> PrepareBaseModelConfig:
        """Internal helper to map YAML to the Entity"""
        model_config = self.config.prepare_base_model[model_name]
        create_directories([model_config.dir])
        
        return PrepareBaseModelConfig(
            root_dir=Path(model_config.dir),
            base_model_path=Path(model_config.raw_path),
            updated_base_model_path=Path(model_config.updated_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_pooling=self.params.POOLING
        )

    # 2. THE EXPLICIT METHODS (The "Interface")
    def get_densenet121_config(self) -> PrepareBaseModelConfig:
        densenet_prepare_config = self._create_config_for('densenet121')
        return densenet_prepare_config

    def get_resnet50_config(self) -> PrepareBaseModelConfig:
        resnet_prepare_config = self._create_config_for('resnet50')
        return resnet_prepare_config

    def get_efficientnetb0_config(self) -> PrepareBaseModelConfig:
        efficientnet_prepare_config = self._create_config_for('efficientnetb0')
        return efficientnet_prepare_config