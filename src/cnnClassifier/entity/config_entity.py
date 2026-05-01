from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzipped_data_dir: Path
    

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    param_image_size: list
    param_learning_rate: float
    param_include_top: bool
    param_weights: str
    param_model_name: str
    params_classes: int
    
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class TrainingConfig:
    root_dir:Path
    trained_model_path:Path
    updated_base_model_path:Path
    training_data: Path
    params_epochs:int
    params_learning_rate:float
    params_batch_size:int
    params_is_augmentation:bool
    params_image_size: list
    
from dataclasses import dataclass
from  pathlib import Path

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: int
    params_batch_size: list

    
