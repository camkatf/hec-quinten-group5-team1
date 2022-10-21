## This notebook with the function to evaluate the model 

import spacy
import random
from spacy.scorer import Scorer
from spacy.training.example import Example
import json


def evaluate(ner_model, test_data):
    """Compute the evaluation scores of the NER SpaCy model.

    Args:
        model : The pipeline to use for scoring.
        data (list): the test data, in a SpaCY JSONL format.

    Returns:
        dic: Scores provided by the individual pipeline components, with indication such as precision, recall and F-score for token character spans.
    """
    #Use Spacy scorer() function
    scorer = Scorer()
    #create an empty dic to store the scores
    examples = []
    for input_, annot in test_data:
        #use the model on the test data
        doc = ner_model.make_doc(input_)
        example = Example.from_dict(doc, annot)
        #make predictions
        example.predicted = ner_model(example.predicted)
        #append results
        examples.append(example)

    return scorer.score(examples)