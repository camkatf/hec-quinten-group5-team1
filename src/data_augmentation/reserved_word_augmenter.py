import logging
logger = logging.getLogger('main_logger')

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
import tqdm
from tqdm import tqdm
import json

# frequency
def augment_frequency(text):
    reserved_tokens_freq = [
        ['/h', 'par heure', '/heure', '/ heure'],
        ['/j', 'par jour ', 'par jour', '/jour', '/ jour', 'jour', 'jours', '/24h', 'jrs'],
        ['hebdomadaire', 'par semaine'], 
        ['q6h', 'toutes les 6 heures', '/6h'], 
        ['q8h', 'toutes les 8 heures', '/8h'],
        ['die', 'x2 jours', 'deux fois par jour', 'bid'],
        ['prn', 'au besoin']
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_freq)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

# drug
def augment_drug(text):
    reserved_tokens_drug = [
        # list of drug from external dataset
        [''],
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_drug)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

# form
def augment_form(text):
    reserved_tokens_form = [
        ['comprimé', 'cps', 'comp', 'capsule'],
        ['ampoules', 'ampoule'], 
        ['narine', 'lunettes nasales'], 
        ['inh', 'inhalation', 'inh : inhalation', 'nébulisation'], 
        ['perfursions', 'perfursion', 'injectable', 'injection']
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_form)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

# route
def augment_route(text):
    reserved_tokens_route = [
        ['intraveineuse', 'intraveineux', 'iv', 'voie iv', 'par voie intraveineuse', 'voie intraveineuse', 'intra-veineuse', 'par voie iv', 'IVL', 'IVD'],
        ['voie orale', 'oraux', 'par la bouche', 'par voie orale', 'oral', 'po', 'po : par la bouche', 'orale', 'per os'], 
        ['transfusions sanguines', 'transfusion', 'perfusion'],
        ['voie systémique', 'systémique'],
        ['voie veineuse périphérique', 'vvp']
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_route)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

# dosage
def augment_dosage(text):
    reserved_tokens_dos = [
        ['deux', '2'], 
        ['trois', '3'],
        ['quatre', '4'], 
        ['1,5', '1', '1,2', '12,8', '2', '1/2', '0,5', '0,1', '1 1/4', '1 1/2'],
        ['4,5', '8,35', '6,25', '37,5', '12'],
        ['10', '20', '30', '40', '50', '60', '70', '80', '90'],
        ['100', '125', '165', '200', '300', '420', '887,5', '160', '800','1000', '1 500'],  
        ['Gy', 'Gray', 'UI', 'unités'], 
        ['1g', 'grammes', 'g', 'milligrammes', 'mg', 'µg', 'mL', 'mEq', 'mcg', 'MUI'],
        ['mg/m2', 'g/m2', 'mg/mL', 'g/L', 'µg/mL', 'ng/l', 'UI/Kg', 'unités/kg' 'mg/kg', 'ml/s', '%']
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_dos)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

# treatment
def augment_treatment(text):
    reserved_tokens_tre = [
        ['soins', 'soin', 'drainage', 'réparation', 'séance', 'épuration', 'thérapie', 'aspiration'], 
        ['perfusion', 'lavage'],
        ['nasal', 'gastrique'],
        ['chirurgie', 'chirurgicale'], 
        ['lombotomie', 'néphrostomie'],
        ['antibiothérapie', 'posologie', 'Médicament'],
        ['radiothérapie', 'chimiothérapie', 'insulinothérapie'], 
        ['antiinflammatoires', 'antibiotiques'],
        ['antibiotique', 'antiseptique', 'antidiabétique', 'antiémétique', 'antituberculeux'], 
        ['poly-', 'bi-', 'radio-'], 
        ['intubées', 'intubée', 'iodées', 'ventilées'], 
        ['intubation', 'sédation'],
        ['oxygénothérapie', 'hydratation', 'réhydratation', 'rééducation'], 
        ['autolyse', 'dialyse', 'hémodialyse'],
        ['actif', 'active', 'totale', 'externe', 'interne', 'droite', 'artificielle', 'artificielle']
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_tre)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

# duration
def augment_dur(text):
    reserved_tokens_dur = [
        ['deux', '2'], 
        ['trois', '3'],
        ['quatre', '4'], 
        ['cinq', '5'], 
        ['six', '6'],
        ['sept', '7'],
        ['huit', '8'], 
        ['neuf', '9'], 
        ['dix', '10'], 
        ['onze', '11'],
        ['douze', '12'], 
        ['quatorze', '14'], 
        ['jours', 'j'], 
        ['heures', 'heure', 'h'], 
        ['36 h', '36 heures', '1 1/2 jours'], 
        ['48 h', '48 heures', '2 jours'], 
        ['24 h', '24 heures', '1 jour'],
        ['/', 'a'],
        ['pendant', 'pour une durée de']
        ]
    reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens_dur)
    augmented_text = reserved_aug.augment(text)

    return augmented_text

def reserved_word_augmenter(labelled_expression, type):
    '''
    Apply the functions to labels depending on their category defined above 
    '''
    if type == 'Frequency':
        transformed= augment_frequency(labelled_expression)
    elif type == 'Dosage':
        transformed = augment_dosage(labelled_expression)
    elif type == 'Drug':
        transformed = augment_drug(labelled_expression)    
    elif type == 'Treatment':    
        transformed = augment_treatment(labelled_expression)
    elif type == 'Form':
        transformed = augment_form(labelled_expression)
    elif type == 'Route':
        transformed = augment_route(labelled_expression)
    elif type == 'Duration':
        transformed = augment_dur(labelled_expression)
    transformed = transformed[0].replace("nnnnn", " ")
    return transformed

    
def process_reserved_word(text, label_position_list):
    '''
    Process back-translation to generate new text.
    Finds the labels in the new text.
    Input : 
        text (str) : text to modify with reserved word augmenter
        label_position_list (list(int, int, str)) : list of the positions and types of the labelled words in the text
    Output : 
        transformed_text (str) : transformed text by modifying the labels with reserved word augmenter
        new_labels (list(int, int, str)) : list of positions and types of the labelled words in the modified text
    '''
    #logger.debug(f'Text : {text[:50]}...')

    #transformed_text = nlpaug_model.augment(text)[0]
    transformed_text = text
    
    #logger.debug(f'Transformed text : {transformed_text[:50]}...')

    if len(label_position_list)==0:
        return transformed_text, []

    # Extract labelled words and modify them with the reserved word augmenter defined per label type previously
    label_list = [text[int(position[0:2][0]): int(position[0:2][1])] for position in label_position_list] 
    
    # Iterrating through the labelled words to change them with the 
    new_labels = label_position_list
    # Keep track of the change of label indices
    diff_accumulated = 0
    for label_ind in range(len(label_list)):  
        type = np.array(label_position_list)[:,2][label_ind]
        label = label_list[label_ind]
        # transform the labels that are defined in the labeled word
        label_transformed = reserved_word_augmenter(label, type)

        # If a labelled word is not modified we pass
        if label == label_transformed:
            #logger.debug('Label not modified.')
            #logger.debug(f"Label: '{label}'.")
            new_labels[label_ind][0] = int(new_labels[label_ind][0]) + diff_accumulated
            new_labels[label_ind][1] = int(new_labels[label_ind][1]) + diff_accumulated
        
        # If a label is modified then we replace it by its modification in the transformed text
        else:
            transformed_text = transformed_text.replace(label, label_transformed)
            # We change the indeces of the label
            #logger.debug(f"Label '{label}' found in reserved word augmenter.")
            new_labels[label_ind][0] = int(new_labels[label_ind][0]) + diff_accumulated
            diff = len(label_transformed)-len(label)
            diff_accumulated = diff_accumulated + diff
            new_labels[label_ind][1] = int(new_labels[label_ind][1]) + diff_accumulated

    return transformed_text, new_labels


def augment_reserved_word_augmentation(df):
    
    for row_id in range(df.shape[0]):
        transformed_text, new_labels = process_reserved_word(df.iloc[row_id]['text'], df.iloc[row_id]['labels'])
        df.iloc[row_id]['text'] = transformed_text
        df.iloc[row_id]['labels'] = new_labels
    
    return df