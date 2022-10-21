##This Notebook to pre-process the test set and test the model.

import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import logging
import json

def from_tokens_to_text(df):
    """
    Preprocessing function: takes a tokenized dataset and recreates sentences.
    Args:
        Data Frame: Tokenized Dataframe

    Returns: 
        Data Frame: Sentenced Dataframe

    """
    texts = []
    #loop through the text.csv file
    for i in df['sentence_id'].unique():
        text = ''
        for idx, token in enumerate(df[df['sentence_id'] == i]['token'].values):
            text = text + token + (" " if idx + 1 < df[df['sentence_id'] == i].shape[0] else "")
        texts.append(text)
    #create DF
    texts = pd.DataFrame(texts)
    texts.columns = ['Texts']
    return texts



def predict_test(prdnlp, test_text):
    """
    Predict function: takes an array of text and performs NER.
    Args:
        Model: the trained model.
        Array: test text.

    Returns: 
        DF: final data frame with tokenId and Prediction
     """
        
    output = []
    
    for i in range(0,60):
        doc = prdnlp(test_text[i])
        labels = []
        for ent in doc.ents:
            labels.append([ent.start_char, ent.end_char, ent.label_]) #ent.end_char,
        for start, end, label in labels:
            span = doc.char_span(start, end, label)
            if span is not None:
                for token in span:
                    token.ent_type_ = label
        for token in doc:
            output.append([i, token.i, token, token.ent_type_])
    
    #Create a DF with the labels predicted and the tokenized text
    results_df = pd.DataFrame(output, columns=["text_id", "token_id", "token", "Predicted"])
    #Add a counter 
    results_df['TokenId'] = range(len(results_df))
    #Replace labels with corresponding IDs
    results_df['Predicted'].replace(to_replace={"Dosage":0, "Drug":1, "Duration":2, "Form":3, "Frequency":4, "": 5, "Route":6, "Treatment":7}, inplace=True)
    #Create Final data Frame for submission
    final_df = results_df[['TokenId', "Predicted"]]
    
    return final_df
    