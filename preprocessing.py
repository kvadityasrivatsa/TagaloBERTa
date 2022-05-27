import os
import re
import csv
import shutil
from time import time
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATA_PATH = './data'
CACHE_BASE_PATH = os.path.join(DATA_PATH,'cache')
CACHE_PATH = os.path.join(CACHE_BASE_PATH,''.join(str(time()).split('.')))
os.makedirs(CACHE_PATH,exist_ok=True)
print("CACHE_PATH: ",CACHE_PATH)

RE_PATTERNS = {'urls':r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",
               'not_alpha_num':r"[^a-zA-Z0-9']",
               }
RE_PATTERNS = {k:re.compile(v) for k,v in RE_PATTERNS.items()}
MAX_SEQ_LEN = 512
MIN_SEQ_LEN = 5

# def parse_csv(path):
#     reader = csv.reader(open(path,'r'))
#     header = next(reader)
#     rows = [r[-1] for r in reader]

def clean_label_df(df):

    texts = df['comment_text']
    texts = [re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",'',s) for s in texts]
    texts = [re.sub(r"[^a-zA-Z0-9']",' ',s) for s in texts]
    texts = [re.sub(r"[ ]+",' ',s) for s in texts]
    texts = [str(s)[:MAX_SEQ_LEN] for s in texts]
    df['comment_text'] = texts

    if 'comment_label' in df:
        labels = df['comment_label']
        labels = [1.0 if int(l)==1 else 0.0 for l in labels]
        labels = [1 if s else 0 for s in labels]
        df['comment_label'] = labels

    return df

def preprocess_label_data(datadf,split=False,test_size=None):
    datadf = datadf.dropna()
    datadf = clean_label_df(datadf)
    datadf['comment_text'] = [str(s)[:MAX_SEQ_LEN].strip() for s in datadf['comment_text']]
    datadf = datadf[[len(l)>MIN_SEQ_LEN for l in datadf['comment_text']]]

    if 'comment_label' in datadf:
        datadf['comment_label'] = [1 if int(l)==1 else 0 for l in datadf['comment_label']]

    datadf = datadf.rename(columns={'comment_id':'comment_id',
                                    'comment_text':'comment_text',
                                    'comment_label':'labels'})
    
    if split:

        print(f'before pruning: {len(datadf)}')
        datadf = datadf.drop_duplicates(subset=['comment_text'],keep='first')
        print(f'after pruning: {len(datadf)}')

        traindf, testdf = train_test_split(datadf,test_size=0.25,random_state=0)
        # traindf = traindf.reset_index(drop=True)
        # testdf = testdf.reset_index(drop=True)
        traindf.to_csv('./data/train.csv',index=False)
        testdf.to_csv('./data/test.csv',index=False)
        dataset = load_dataset('csv', data_files={'train':'./data/train.csv',
                                                  'test':'./data/test.csv'},
                                    cache_dir=CACHE_PATH)

    else:
        # datadf = datadf.reset_index(drop=True)
        datadf.to_csv('./data/data.csv',index=False)
        dataset = load_dataset('csv',data_files={'eval':'./data/data.csv'},
                                cache_dir=CACHE_PATH)
    return dataset, datadf

def linear_join(srcdf,predf):
    '''
        srcdf: ["comment_id","comment_text"]
        predf: ["comment_id","comment_label"]
    '''
    predf_lookup = {r['comment_id']:int(r['comment_label']) for _,r in tqdm(predf.iterrows(),total=len(predf))}
    srcdf['comment_label'] = [predf_lookup[r['comment_id']] if r['comment_id'] in predf_lookup else 0 for _,r in tqdm(srcdf.iterrows(),total=len(srcdf))]
    return srcdf

def clear_cache():
    shutil.rmtree(CACHE_PATH)
    print(f"CACHE_PATH: {CACHE_PATH} cleared.")