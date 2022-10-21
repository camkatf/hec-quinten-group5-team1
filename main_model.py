import pandas as pd
import os
import yaml

from src.Utils.argparser import get_args
from src.Utils.utils import my_get_logger
from src.model.data_preprocessing import *
from src.model.train import *
from src.model.evaluate import *
from src.model.testing import *
warnings.simplefilter("ignore")

# Getting config
args = get_args()
config_path = args.get('config_path')
with open(config_path, "r") as file:
    config_dict = yaml.safe_load(file)

# Instanciation of the logger
path_log = config_dict.get("PATH_LOG")
log_level = config_dict.get("LOG_LEVEL")
logger = my_get_logger(path_log, log_level, my_name="bt_logger")


def main(logger, config : dict):
    input_path = config.get("INPUT_PATH")
    output_path = config.get("OUPTUT_PATH_DATA_AUG")
    from_model_name = config.get('FROM_MODEL_NAME')
    to_model_name = config.get('TO_MODEL_NAME')
    text_column = config.get('TEXT_COLUMN')
    label_column = config.get('LABEL_COLUMN')
    column_names = list(config.get("COLUMN_NAMES"))

    logger.info("Start of the pipeline.")
    # Import datasets
    test_df = pd.read_csv("data/test.csv")
    logger.info("Test set imported.")

    #Train data
    DATA_FILE_PATH = input_path
    ALL_DATA = fillterDoccanoData(DATA_FILE_PATH)
    ALL_DATA=trim_entity_spans(ALL_DATA)
    ALL_DATA = validate_overlap(ALL_DATA)  

    # Train model
    prdnlp = train_spacy(ALL_DATA, 1)

    # Predict on test data
    test_data = from_tokens_to_text(test_df)
    test_text = test_data["Texts"]
    predict_test(prdnlp, test_text).to_csv(os.path.join(output_path, "predictions.csv"), index=False, header=True)

    return 


if __name__ == '__main__':
    try:
        main(logger, config_dict)
    
    except Exception as e:
        logger.error("Error during execution", exc_info=True)