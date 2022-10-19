import pandas as pd
import os
import yaml
import src.Utils as u
from src.Utils.argparser import get_args
import nlpaug.augmenter.word as naw
from src.back_translation import process

# Getting config
args = get_args()
config_path = args.get('config_path')
with open(config_path, "r") as file:
    config_dict = yaml.safe_load(file)

# Instanciation of the logger
path_log = config_dict.get("PATH_LOG")
log_level = config_dict.get("LOG_LEVEL")
logger = u.my_get_logger(path_log, log_level, my_name="bt_logger")


def main(logger, config : dict):
    input_path = config.get("INPUT_PATH")
    data_path = config.get("DATA_PATH")
    from_model_name = config.get('FROM_MODEL_NAME')
    to_model_name = config.get('TO_MODEL_NAME')
    text_column = config.get('TEXT_COLUMN')
    label_column = config.get('LABEL_COLUMN')

    logger.info("Start of the pipeline.")

    # Read data
    df = pd.read_csv(input_path).loc[:1, :]
    logger.info("Data read.")

    # Load back-translation model
    back_translation_aug = naw.BackTranslationAug(
        from_model_name=from_model_name, 
        to_model_name=to_model_name
    )
    logger.info("Back-translation model loaded.")

    # Process back-translation
    result = df[[text_column, label_column]].apply(lambda x: process(x[0], x[1], back_translation_aug), axis=1)
    logger.info("Back-translation ended.")

    # Save results
    df[["transformed_text", "new_labels"]] = pd.DataFrame(result.to_list())
    df.to_csv(os.path.join(data_path, 'data_with_bt.csv'), header=True, index=False)
    logger.info("Results saved.")

    return df.head()


if __name__ == '__main__':
    try:
        main(logger, config_dict)
    
    except Exception as e:
        logger.error("Error during execution", exc_info=True)