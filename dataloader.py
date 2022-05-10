import os
import gdown
import shutil
import pandas as pd
from datasets import load_dataset

DATA_PATH = './data'
BASE_MODEL_PATH = os.path.join(DATA_PATH,'base_models')
FINETUNING_DATA_PATH = os.path.join(DATA_PATH,'finetuning_data')
FINTUNED_MODEL_PATH = os.path.join(DATA_PATH,'fintuned_models')

model_urls = {'TagaloBERTa_1M':'https://drive.google.com/uc?id=199_fviLladsuamgmUwWSA0ax1sb21EYT',
            'TagaloBERTa_10M':'https://drive.google.com/uc?id=1XJBQSfK3KyLC9lUHqley85MITVikFLjb',
            'TagaloBERTa_30M':'https://drive.google.com/uc?id=1LI-nudLwAA1pTUoNhS_dVwsDvjPGK_Dg',
            }

custom_data_urls = {'balanced.csv':None,
                    'balanced_iterative_5.csv':'https://drive.google.com/uc?id=1OF99m9Q8BsolWctCbRRNCYMVksVFFnvw',
                    'balanced_iterative_10.csv':'https://drive.google.com/uc?id=1gpBm5bvvIJKD_cUdp7e0hVNMjaoi4dHT',
                    'balanced_iterative_20.csv':'https://drive.google.com/uc?id=1ZcwvV0CHjRg1kv8ogXfkVGf4vIKHuW85',
                    'balanced_iterative_30.csv':'https://drive.google.com/uc?id=1VONn7GToySkbdiwy8HgG3djUdufqdmjQ',
                    }

def fetch_base_encoder(model):
    if model not in model_urls:
        raise Exception(f"Invalid base model: '{model}' is not available. Select one of the following: {list(model_urls.keys())}")
    else:
        base_path = os.path.join(BASE_MODEL_PATH,model+'.zip')
        gdown.cached_download(model_urls[model],base_path)
        shutil.unpack_archive(base_path,BASE_MODEL_PATH,'zip')
        return os.path.join(BASE_MODEL_PATH,model)

def load_custom_data(path,col_map={'text':'comment_text','labels':'annotation'}):
    if path not in custom_data_urls:
        raise Exception(f"Invalid dataset: '{path}' is not available. Select one of the following: {list(custom_data_urls.keys())}")
    else:
        data_path = os.path.join(FINETUNING_DATA_PATH,path)
        gdown.cached_download(custom_data_urls[path],data_path)
        datadf = pd.read_csv(data_path)
        datadf = datadf.rename(columns={v:k for k,v in col_map.items()})
    return datadf

def load_huggingface_data(path,split='test',col_map={'text':'text','labels':'label'}):
    data = load_dataset(path)
    datadf = pd.DataFrame({'text':data[split][col_map['text']],'labels':data[split][col_map['labels']]})
    return datadf


