## This Notebook with the SpaCy model and cleaned output for submission.

import spacy
import random
import time
import warnings
from spacy.util import minibatch, compounding, decaying
from spacy.training.example import Example

#Define the model
model = "fr_core_news_lg"

#Function to train the model
def train_spacy(data, iterations):
    """Train a NER SpaCy model.

    Args:
        data (list): The cleaned data.
        iterations: The number of iterations to train the model.

    Returns:
        list: the token's label, the starting position of the label, the ending position of the label.
    """
    #load the model
    nlp = spacy.load(model)
    
    #define some special case to tokenize the data similarly to the test.csv
    nlp.tokenizer.add_special_case(u"d'", [{"ORTH": u"d'"}])
    nlp.tokenizer.add_special_case(u"D'", [{"ORTH": u"D'"}])
    nlp.tokenizer.add_special_case(u"d’", [{"ORTH": u"d’"}])
    nlp.tokenizer.add_special_case(u"D’", [{"ORTH": u"D’"}])
    nlp.tokenizer.add_special_case(u"l'", [{"ORTH": u"l'"}])
    nlp.tokenizer.add_special_case(u"L'", [{"ORTH": u"L'"}])
    nlp.tokenizer.add_special_case(u"l’", [{"ORTH": u"l’"}])
    nlp.tokenizer.add_special_case(u"L’", [{"ORTH": u"L’"}])
    nlp.tokenizer.add_special_case(u"j'", [{"ORTH": u"j'"}])
    nlp.tokenizer.add_special_case(u"J'", [{"ORTH": u"J'"}])
    nlp.tokenizer.add_special_case(u"j’", [{"ORTH": u"j’"}])
    nlp.tokenizer.add_special_case(u"J’", [{"ORTH": u"J’"}])
    nlp.tokenizer.add_special_case(u"c'", [{"ORTH": u"c'"}])
    nlp.tokenizer.add_special_case(u"C'", [{"ORTH": u"C'"}])
    nlp.tokenizer.add_special_case(u"c’", [{"ORTH": u"c’"}])
    nlp.tokenizer.add_special_case(u"C’", [{"ORTH": u"C’"}])
    nlp.tokenizer.add_special_case(u"s'", [{"ORTH": u"s'"}])
    nlp.tokenizer.add_special_case(u"S'", [{"ORTH": u"S'"}])
    nlp.tokenizer.add_special_case(u"s’", [{"ORTH": u"s’"}])
    nlp.tokenizer.add_special_case(u"S’", [{"ORTH": u"S’"}])
    nlp.tokenizer.add_special_case(u"n'", [{"ORTH": u"n'"}])
    nlp.tokenizer.add_special_case(u"N'", [{"ORTH": u"N'"}])
    nlp.tokenizer.add_special_case(u"n’", [{"ORTH": u"n’"}])
    nlp.tokenizer.add_special_case(u"N’", [{"ORTH": u"N’"}])
    nlp.tokenizer.add_special_case(u"qu'", [{"ORTH": u"qu'"}])
    nlp.tokenizer.add_special_case(u"Qu'", [{"ORTH": u"Qu'"}])
    nlp.tokenizer.add_special_case(u"qu’", [{"ORTH": u"qu’"}])
    nlp.tokenizer.add_special_case(u"Qu’", [{"ORTH": u"Qu’"}])
    nlp.tokenizer.add_special_case(u"puisqu'", [{"ORTH": u"puisqu'"}])
    nlp.tokenizer.add_special_case(u"Puisqu'", [{"ORTH": u"Puisqu'"}])
    nlp.tokenizer.add_special_case(u"puisqu’", [{"ORTH": u"puisqu’"}])
    nlp.tokenizer.add_special_case(u"Puisqu’", [{"ORTH": u"Puisqu’"}])
    nlp.tokenizer.add_special_case(u"jusqu'", [{"ORTH": u"jusqu'"}])
    nlp.tokenizer.add_special_case(u"Jusqu'", [{"ORTH": u"Jusqu'"}])
    nlp.tokenizer.add_special_case(u"jusqu’", [{"ORTH": u"jusqu’"}])
    nlp.tokenizer.add_special_case(u"Jusqu’", [{"ORTH": u"Jusqu’"}])
    suffixes = [ss for ss in nlp.Defaults.suffixes if 'G' not in ss]
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search

    
    print("Loaded model '%s'" % model)
    
    TRAIN_DATA = data
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    else:
        ner = nlp.get_pipe("ner")
      
    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    if model is None:
        optimizer = nlp.begin_training()
      
        # For training with customized cfg 
        nlp.entity.cfg['conv_depth'] = 16
        nlp.entity.cfg['token_vector_width'] = 256
      
    else:
        print ("resuming")
        optimizer = nlp.resume_training()
        print(optimizer.learn_rate)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    dropout = decaying(0.8, 0.2) #minimum, max, decay rate
    sizes = compounding(1.0, 4.0, 1.001)

    with nlp.disable_pipes(*other_pipes):  
      # only train NER
      for itn in range(iterations):
        print("Starting iteration " + str(itn))
        random.shuffle(TRAIN_DATA)
        losses = {}

        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
              [example],
              drop=0.2,  # dropout - make it harder to memorise data
              sgd=optimizer,  # callable to update weights
              losses=losses)
        print(losses)
    return nlp