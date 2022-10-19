
# -*- coding: utf-8 -*-
import json
from stat import filemode
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
logger = logging.getLogger('bt_logger')

def my_get_logger(path_log, log_level, my_name =""):
    """
    Instanciation du logger et paramétrisation
    :param path_log: chemin du fichier de log
    :param log_level: Niveau du log
    :return: Fichier de log
    """
    log_level_dict = {"CRITICAL": logging.CRITICAL,
                        "ERROR": logging.ERROR,
                        "WARNING": logging.WARNING,
                        "INFO": logging.INFO,
                        "DEBUG": logging.DEBUG}
    
    LOG_LEVEL = log_level_dict[log_level]

    if my_name != "":
        logger = logging.getLogger(my_name)
        logger.setLevel(LOG_LEVEL)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(LOG_LEVEL)
    
    # create a file handler
    handler = logging.FileHandler(path_log)
    #, mode="w"
    handler.setLevel(LOG_LEVEL)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)-8s: %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def read_jsonl_df(input_path: str, column_names: list) -> pd.DataFrame:
    '''
    Read .jsonl input and convert it to pd.dataframe
    '''
    with open(input_path, 'r') as json_file:
        json_list = list(json_file)
    
    list_columns = []
    for n, val in enumerate(column_names):
        globals()[f"{val}"] = []
        list_columns += [globals()[f"{val}"]]

    print("Converting jsonl file :")
    for json_str in tqdm(json_list):
        result = json.loads(json_str)

        for i in range(len(list_columns)):
            list_columns[i] += [result[column_names[i]]]

    df = pd.DataFrame(list_columns).T
    df.columns = column_names

    return df