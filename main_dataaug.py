import pandas as pd
import os
import yaml
from tqdm import tqdm

from src.Utils.argparser import get_args
import nlpaug.augmenter.word as naw
from src.data_augmentation.back_translation import process
from src.Utils.utils import read_jsonl_df, my_get_logger
from src.data_augmentation.reserved_word_augmenter import *

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

    # Read data
    df = read_jsonl_df(input_path, column_names)
    logger.info("Data read.")

    # Reserved word augmentation
    df_aug = augment_reserved_word_augmentation(df[~df['labels'].isin([[]])])
    df = read_jsonl_df(input_path, column_names)
    df_augmented = df.append(df_aug).reset_index()
    logger.info("Data augmented with reserved word method.")

    # Load back-translation model
    back_translation_aug = naw.BackTranslationAug(
        from_model_name=from_model_name, 
        to_model_name=to_model_name
    )
    logger.info("Back-translation model loaded.")

    # Process back-translation
    tqdm.pandas()
    print("Applying back-translation :")
    result = df_augmented[[text_column, label_column]].progress_apply(lambda x: process(x[0], x[1], back_translation_aug), axis=1)
    logger.info("Back-translation ended.")

    # Save results
    df_augmented[["transformed_text", "new_labels"]] = pd.DataFrame(result.to_list())
    df_augmented.to_csv(os.path.join(output_path, 'data_with_bt.csv'), header=True, index=False)
    logger.info("Results saved.")

    return df_augmented.head()


if __name__ == '__main__':
    try:
        main(logger, config_dict)
    
    except Exception as e:
        logger.error("Error during execution", exc_info=True)