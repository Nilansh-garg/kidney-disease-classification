import zipfile
import gdown
import os
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.config.configuration import ConfigurationManager

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self)-> str:
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok = True)
            logger.info(f"downloading data from {dataset_url} into file {zip_download_dir}")
            
            file_id = dataset_url.split('/')[-2]
            gdown.download(id=file_id, output=zip_download_dir)

            logger.info(f"file downloaded successfully and saved at {zip_download_dir} with size {get_size(Path(zip_download_dir))}")
            return zip_download_dir
        except Exception as e:
            raise e
    
    
    def extract_zip_file(self, zip_file_path: str):
        try:
            logger.info(f"extraction of file: {zip_file_path} started")
            unzip_dir = self.config.unzipped_data_dir
            os.makedirs(unzip_dir, exist_ok = True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            logger.info(f"extraction completed successfully at location: {unzip_dir}")
        except Exception as e:
            raise e
        
        