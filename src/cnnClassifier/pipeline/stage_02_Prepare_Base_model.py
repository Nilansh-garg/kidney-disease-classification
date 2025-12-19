import gc
import tensorflow as tf
from cnnClassifier import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import prepareBaseModel

STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        
        # Define the models we want to iterate through
        model_tasks = [
            (config_manager.get_densenet121_config, "densenet121"),
            (config_manager.get_resnet50_config, "resnet50"),
            (config_manager.get_efficientnetb0_config, "efficientnetb0")
        ]

        for get_config_method, model_name in model_tasks:
            try:
                logger.info(f">>>>>> Starting task for {model_name} <<<<<<")
                
                # 1. Clear Keras global state and run Garbage Collector
                tf.keras.backend.clear_session()
                gc.collect()
                
                # 2. Initialize Configuration and Component
                config = get_config_method()
                prepare_base_model = prepareBaseModel(config=config)
                
                # 3. Download and Prepare
                prepare_base_model.get_base_model(model_type=model_name)
                prepare_base_model.update_base_model()
                
                # 4. Cleanup internal references to free RAM immediately
                del prepare_base_model
                del config
                
                logger.info(f">>>>>> {model_name} completed successfully <<<<<<\n")
                
            except Exception as e:
                logger.error(f"Error occurred during {model_name} preparation: {e}")

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e