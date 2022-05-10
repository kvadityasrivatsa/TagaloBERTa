import os
import gdown
import shutil
import pandas as pd
from datasets import load_dataset

DATA_PATH = './data'
BASE_MODEL_PATH = os.path.join(DATA_PATH,'base_models')
FINTUNED_MODEL_PATH = os.path.join(DATA_PATH,'fintuned_models')

model_urls = {'TagaloBERTa_1M':'https://drive.google.com/uc?id=199_fviLladsuamgmUwWSA0ax1sb21EYT',
            'TagaloBERTa_10M':'https://drive.google.com/uc?id=1XJBQSfK3KyLC9lUHqley85MITVikFLjb',
            'TagaloBERTa_30M':'https://drive.google.com/uc?id=1LI-nudLwAA1pTUoNhS_dVwsDvjPGK_Dg',
            }

def fetch_base_encoder(model):
    if model not in model_urls:
        raise Exception(f'Invalid base model: {model} is not available. Select one of the following: {list(model_urls.keys())}')
    else:
        base_path = os.path.join(BASE_MODEL_PATH,model+'.zip')
        gdown.cached_download(model_urls[model],base_path)
        shutil.unpack_archive(base_path,BASE_MODEL_PATH,'zip')
        return os.path.join(BASE_MODEL_PATH,model)

def load_custom_data(path,col_map={'text':'comment_text','labels':'annotation'}):
    datadf = pd.read_csv(path)
    datadf = datadf.rename(columns={v:k for k,v in col_map.items()})
    return datadf

def load_huggingface_data(path,split='test',col_map={'text':'text','labels':'label'}):
    data = load_dataset(path)
    datadf = pd.DataFrame({'text':data[split][col_map['text']],'labels':data[split][col_map['labels']]})
    return datadf


