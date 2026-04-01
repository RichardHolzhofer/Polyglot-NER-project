import logging
import os

LOG_FILE = "polyglot_ner.log"

logs_folder_path = os.path.join(os.getcwd(), "logs")

os.makedirs(logs_folder_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_folder_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
