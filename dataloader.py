import os
import re
import csv
import gdown
import shutil
import pandas as pd
import pickle
from tqdm import tqdm
from datasets import load_dataset

DATA_PATH = './data'
CACHE_BASE_PATH = os.path.join(DATA_PATH,'cache')
FINETUNING_DATA_PATH = os.path.join(DATA_PATH,'finetuning_data')
BASE_MODEL_PATH = os.path.join(DATA_PATH,'base_models')
FINETUNED_MODEL_PATH = os.path.join(DATA_PATH,'fintuned_models')

os.makedirs(DATA_PATH,exist_ok=True)
os.makedirs(CACHE_BASE_PATH,exist_ok=True)

base_model_urls = {'TagaloBERTa_1M':'https://drive.google.com/uc?id=199_fviLladsuamgmUwWSA0ax1sb21EYT',
            'TagaloBERTa_10M':'https://drive.google.com/uc?id=1XJBQSfK3KyLC9lUHqley85MITVikFLjb',
            'TagaloBERTa_30M':'https://drive.google.com/uc?id=1LI-nudLwAA1pTUoNhS_dVwsDvjPGK_Dg',
            }

finetuned_model_urls = {'TagaloBERTa_RoBERTa_Bi10_30M.model':'https://drive.google.com/uc?id=1SiOXy7ma1nQckQKJ249pNnSzFh8FmhCE',
                        'TagaloBERTa_intersection_30M.model':'https://drive.google.com/uc?id=12iFanE1a9l6fU9YhHsuVxUqJKJTY9879',
                        'TagaloBERTa_hgfc_30M.model':'https://drive.google.com/uc?id=1QjZVSKG5H03jLppZEVH9V6h0EuI5pb24',
                        'TagaloBERTa_Bi10_30M.model':'https://drive.google.com/uc?id=13KizqMchhlALLAWOm2t9igmCbEbka2ga',
                        'TagaloBERTa_Bi20_30M.model':'https://drive.google.com/uc?id=1K1e9H8MV0uTGbVal4cZ3OUbwGXtdHxuH',
                        'TagaloBERTa_RoBERTa_hsf_30M.model':'https://drive.google.com/uc?id=1JozmqAIFE8pr5UeL6IYc4BNhpj-ep8NL',
                        'TagaloBERTa_hgfc_plus_Bi10_30M.model':'https://drive.google.com/uc?id=1GBI1GkSQvz5Hk8EOR7GFI2NGxY3rxETS',
                        'rf_model':'https://drive.google.com/uc?id=1-7AZZZhL3M_ZUY1b23Lk4BzFD50q4G1A',
                    }

custom_data_urls = {'balanced.csv':None,
                    'balanced_iterative_5.csv':'https://drive.google.com/uc?id=1OF99m9Q8BsolWctCbRRNCYMVksVFFnvw',
                    'balanced_iterative_10.csv':'https://drive.google.com/uc?id=1zP5ghhJV15j_opHX9-uOcC_XaQV2M5s7',
                    'balanced_iterative_20.csv':'https://drive.google.com/uc?id=1ZcwvV0CHjRg1kv8ogXfkVGf4vIKHuW85',
                    'balanced_iterative_30.csv':'https://drive.google.com/uc?id=1VONn7GToySkbdiwy8HgG3djUdufqdmjQ',
                    'hgfc_plus_Bi10.csv':'https://drive.google.com/uc?id=1kjozHPdEnaw_4wbTMJdLsFoAUFcCe5LN',
                    'intersection_data.csv':'https://drive.google.com/uc?id=1kylIL2IE-d29nmy2iSqLeoi-0qxyzdAx',
                    'intersection_labelled.csv':'https://drive.google.com/uc?id=1rQDc21kzQexf-b176HwTawUC0SHU2wRM',
                    }

def fetch_base_model(model):
    if model not in base_model_urls:
        raise Exception(f"Invalid base model: '{model}' is not available. Select one of the following: {list(base_model_urls.keys())}")
    else:
        os.makedirs(BASE_MODEL_PATH,exist_ok=True)
        base_path = os.path.join(BASE_MODEL_PATH,model+'.zip')
        gdown.cached_download(base_model_urls[model],base_path)
        shutil.unpack_archive(base_path,BASE_MODEL_PATH,'zip')
        return os.path.join(BASE_MODEL_PATH,model)

def fetch_finetuned_model(model):
    if model not in finetuned_model_urls:
        raise Exception(f"Invalid finetuned model: '{model}' is not available. Select one of the following: {list(finetuned_model_urls.keys())}")
    else:
        os.makedirs(FINETUNED_MODEL_PATH,exist_ok=True)
        finetuned_path = os.path.join(FINETUNED_MODEL_PATH,model+'.zip')
        gdown.cached_download(finetuned_model_urls[model],finetuned_path)
        shutil.unpack_archive(finetuned_path,FINETUNED_MODEL_PATH,'zip')
        return os.path.join(FINETUNED_MODEL_PATH,model)

# def load_finetuning_data(path,col_map={'text':'comment_text','labels':'comment_label'}):
#     if path not in finetuning_data_urls:
#         raise Exception(f"Invalid dataset: '{path}' is not available. Select one of the following: {list(finetuning_data_urls.keys())}")
#     else:
#         data_path = os.path.join(FINETUNING_DATA_PATH,path)
#         gdown.cached_download(finetuning_data_urls[path],data_path)
#         datadf = pd.read_csv(data_path)
#         # datadf = datadf.rename(columns={v:k for k,v in col_map.items()}).reset_index(drop=True)
#     return datadf

def parse_csv(path):
    print(f'parsing {path}')
    reader = csv.reader(open(path,'r'))
    header = next(reader)
    rows = [r for r in reader]
    try:
        id_col, text_col = header.index('comment_id'), header.index('comment_text')
    except:
        raise Exception('mandatory columns not found.')
    id_list, text_list = zip(*[[rows[i][id_col],rows[i][text_col]] for i in tqdm(range(len(rows))) if len(rows[i])==len(header)])
    df = pd.DataFrame({'comment_id':id_list,'comment_text':text_list})
    print(f'parser: {len(df)} raw examples.')
    return df

def load_external_data(path):
    if not os.path.exists(path):
        raise Exception(f'File not found. {path}')
    else:
        # datadf = pd.read_csv(path)
        datadf = parse_csv(path)
        if all([c in datadf.columns for c in ['comment_id','comment_text']]):
            return datadf
        else:
            raise Exception('File must atleast contain the following columns: ["comment_id","comment_text"]')

def load_huggingface_data(path,split='test',col_map={'text':'comment_text','labels':'comment_label'}):
    try:
        data = load_dataset(path)
    except:
        return False, None
    datadf = pd.DataFrame({'comment_text':data[split][col_map['text']],'label':data[split][col_map['labels']]})
    return True, datadf

def load_rf_model():
    os.makedirs(FINETUNED_MODEL_PATH,exist_ok=True)
    rf_model_path = os.path.join(FINETUNED_MODEL_PATH,'rf_model.zip')
    gdown.cached_download(finetuned_model_urls['rf_model'],rf_model_path)
    with open(rf_model_path,'rb') as f:
        model = pickle.load(f)
    def rf_cleaner(l):
        l = [re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",'',s if isinstance(s,str) else '') for s in l]
        l = [re.sub(r"[^a-z0-9']",' ',s.lower()) for s in l]
        l = [re.sub(r"[ ]+",' ',s.lower()) for s in l]
        return l
    return model['rf_model'], model['rf_vectorizer'], rf_cleaner