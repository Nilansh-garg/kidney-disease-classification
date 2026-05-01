from cnnClassifier.config.configurartion import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

STAGE_NAME = "EVALUATION STAGE"

class EvaluationPipeline:
    try:
        def __init__(self):
            pass
        
        def main(self):
            config = ConfigurationManager()
            eval_config = config.get_validation_config()
            evaluation = Evaluation(config=eval_config)
            evaluation.evaluation()
            evaluation.save_score()
            evaluation.log_into_mlflow()
    except Exception as e:
        raise e
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>stage {STAGE_NAME} started<<<<<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    