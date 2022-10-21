# Name entity recognition of medical report
This repository aims at spotting automatically the main elements of medical reports.  
The labels concerned are the following :
- Treatment
- Drug
- Duration
- Frequency
- Dosage
- Route
- Form
  
To install all required packages, please run the command `pip install -r requirements.txt` in the terminal.  
The configuration is to be set in the `config.yml` file.  
  
## First pipeline : data augmentation
This pipeline augments the data of the `INPUT_PATH` file, using reserved words and back-translation.  
To run this pipeline, run the two following commands:  
- `python -m spacy download fr_core_news_lg`  
- `python main_model.py`  
The resulting dataset is saved at `OUPTUT_PATH_DATA_AUG`.  
  
## Second pipeline : model training and prediction
This pipeline trains a model and labels the texts in the file located at `INPUT_PATH`.  
To run this pipeline, please run the command `python main_bt_.py`.  
The resulting dataset is saved at `OUPTUT_PATH_MODEL`.