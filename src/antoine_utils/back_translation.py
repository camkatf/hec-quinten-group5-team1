import numpy as np
import pandas as pd
import re
import logging
logger = logging.getLogger('bt_logger')

def find_position(word, text, type):
    '''
    Finds the positions of all the occurences of the word in the text.

    Input : 
        word (str) : word to find
        text (str) : text were to search the word in
        type (str) : label of the word to find
    Output : 
        (list(list(int, int, str))) : list of the position of each labelled word in the text
    '''
    word_length = len(word)
    first_letter_positions = [m.start() for m in re.finditer(word, text)]
    return [[start, start+word_length, type] for start in first_letter_positions]


def process(text, label_position_list, nlpaug_model):
    '''
    Process back-translation to generate new text.
    Finds the labels in the new text.

    Input : 
        text (str) : text to back-translate
        label_position_list (list(int, int, str)) : list of the positions and types of the labelled words in the text
        nlpaug_model : nlpaug model already trained on a specific language
    Output : 
        transformed_text (str) : back-translated text
        new_labels (list(int, int, str)) : list of positions and types of the labelled words in the back-translated text
    '''
    logger.debug(f'Text : {text[:50]}...')

    transformed_text= nlpaug_model.augment(text)[0]
    logger.debug(f'Transformed text : {transformed_text[:50]}...')

    if len(label_position_list)==0:
        return transformed_text, []

    # Extract labelled words and back-translate them
    label_list = [text[int(position[0:2][0]): int(position[0:2][1])] for position in np.array(label_position_list)[0:1]] 
    transformed_labels = nlpaug_model.augment(label_list)

    if transformed_text==text:
        logger.debug(f'Back-translation did not change the text.')
        return None, None
    
    # Iterrating through the labelled word to see if we can find them in the back-translated text
    new_labels = []
    for label_ind in range(len(label_list)):   
        type = np.array(label_position_list)[:,2][label_ind]
        
        label = label_list[label_ind]
        label_transformed = transformed_labels[label_ind] 
        
        # If a labelled word is lost, we drop the back-translation
        if label not in transformed_text and label_transformed not in transformed_text :
            logger.debug('Label lost in translation.')
            logger.debug(f"Label: '{label}'.")
            logger.debug(f"Transformed label: '{label_transformed}'.")
            return None, None

        elif label in transformed_text :
            logger.debug(f"Label '{label}' found in transformed text.")
            new_positions = find_position(label_list[label_ind], transformed_text, type)
            new_labels += new_positions

        elif label_transformed in transformed_text:
            logger.debug(f"Transformed label '{label_transformed}' found in transformed text.")
            new_positions = find_position(transformed_labels[label_ind], transformed_text, type)
            new_labels += new_positions

    return transformed_text, new_labels