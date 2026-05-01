from cnnClassifier.config.configurartion import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger 

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config
        
        self.training_config = config.get_training_config()
        self.training = Training(config = self.training_config)
        
    def run(self):
        logger.info(f"********************************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")

        # --- AGGRESSIVE DEBUGGING CHECK ---
        data_path = self.training_config.training_data
        base_model_path = self.training_config.updated_base_model_path
        
        # 1. Check Data Path
        if not data_path.exists():
            logger.error(f"FATAL ERROR: Training data directory not found at: {data_path}")
            # Raise an error to stop DVC and show the problem
            raise FileNotFoundError(f"Training data directory not found at: {data_path}")

        # 2. Check Base Model Path
        if not base_model_path.exists():
            logger.error(f"FATAL ERROR: Base Model not found at: {base_model_path}")
            raise FileNotFoundError(f"Base Model not found at: {base_model_path}")
        
        logger.info("--- DEBUG: All input paths confirmed to exist! ---")
        # ----------------------------------
        
        self.training.get_base_model()
        self.training.train_valid_generator()
        self.training.train()
        
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<")
        
from cnnClassifier.config.configurartion import ConfigurationManager
from cnnClassifier.components.model_training import Training
        
        
if __name__ == '__main__':
    try:
        print("--- DEBUG CHECKPOINT 1: Starting script execution. ---")
        
        print("--- DEBUG CHECKPOINT 2: All imports successful. ---")
        
        config = ConfigurationManager()
        print("--- DEBUG CHECKPOINT 3: ConfigurationManager instantiated. ---")

        obj = ModelTrainingPipeline(config = config)
        print("--- DEBUG CHECKPOINT 4: ModelTrainingPipeline instantiated. ---")

        obj.run()
        
    except Exception as e:
        print(f"--- FATAL ERROR: Script crashed during initialization! Exception: {e} ---")
        logger.exception(e)
        raise e