## This notebook to pre-process the data extracted from doccano and transform it into a .spacy format

from string import punctuation
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import logging
import json
import re


def fillterDoccanoData(doccano_JSONL_FilePath):
    """Transformes Doccano data .

    Args:
        data file path (JSONL): file path to the extracted Doccano data, in JSONL format.

    Returns:
        list: The training dataset, in SpaCy JSON format.
    """
    try:
        training_data = []
        lines=[]
        with open(doccano_JSONL_FilePath, 'r', encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['text']
            entities = data['labels']
            training_data.append((text, {"entities" : entities}))
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + doccano_JSONL_FilePath + "\n" + "error = " + str(e))
        return None
    
    
def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data

def validate_overlap(ALL_DATA):
    """Validates the data in correct format for training.

    Args:
        data (list): The cleaned data.

    Returns:
        list: The cleaned data, validates.
    """
    for ix,x in enumerate(ALL_DATA):
        startCK=[]
        for iy,y in enumerate(x[-1]['entities']):
            if iy == 0:
                startCK.append([y[0],y[1]])
            else:
                pop = False 
                for z in startCK:
                    if z[0] <= y[0] < z[1]:
                        print(y,z)
                        ALL_DATA[ix][-1]['entities'].pop(iy)
                        print(ALL_DATA[ix][-1]['entities'].pop(iy))
                        pop = True
                        break
                if pop == False:
                    startCK.append([y[0],y[1]])
    return ALL_DATA