import warnings
warnings.filterwarnings("ignore")

import pdb

import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from dataloader import fetch_base_model, fetch_finetuned_model
from dataloader import load_huggingface_data, load_external_data
from preprocessing import preprocess_label_data, linear_join, clear_cache

argp = argparse.ArgumentParser()
argp.add_argument('--raw-data',type=str,dest='rpath',required=True,default=None)
argp.add_argument('--output-path',type=str,dest='xpath',required=True,default='output.csv')
argp.add_argument('--tuned-model',type=str,dest='tpath',default='TagaloBERTa_Bi10_30M.model')
argp.add_argument('--base-model',type=str,dest='bpath',default='TagaloBERTa_30M')
argp.add_argument('--tuned-model-split',type=str,dest='tsplit',default='test')
args = argp.parse_args()

print('loading data to be labelled.')
res, rawdf = load_huggingface_data(args.rpath,args.tsplit)
if not res:
    # datadf = load_finetuning_data(args.rpath)
    rawdf = load_external_data(args.rpath)

# pdb.set_trace(header='post rawdf loading')

rawdf = rawdf.dropna(subset=['comment_id'])
rawdf = rawdf[['comment_id','comment_text','comment_label'] if 'comment_label' in rawdf.columns else ['comment_id','comment_text']]
rawdf['comment_id'] = rawdf['comment_id'].astype(int)
# rawdf = rawdf.set_index('comment_id')

# pdb.set_trace(header='post prelim row drop') j

print('preprocessing data.')
dataset, datadf = preprocess_label_data(rawdf.copy(),split=False)

# pdb.set_trace(header='port preproc')

print('tokenizing data.')
base_model_path = fetch_base_model(args.bpath)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
def tokenize_function(examples):
    return tokenizer(examples["comment_text"], 
                     padding="max_length", 
                     truncation=True,
                     max_length=512)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print('loading model.')
model_args = TrainingArguments(output_dir='./data/model')
finetuned_model_path = fetch_finetuned_model(args.tpath)
model = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(finetuned_model_path,num_labels=2),
    args=model_args,
)

print('generating labels.')
_pred = model.predict(tokenized_dataset['eval'])
pred = np.argmax(_pred.predictions,axis=1)
if 'labels' in datadf:
    print(classification_report(pred,tokenized_dataset['eval']['labels']))
datadf['comment_label'] = list(pred)

# pdb.set_trace(header='pre join')

# rawdf = rawdf.join(datadf,on='comment_id',lsuffix='',rsuffix='_R')
# rawdf['comment_label'] = rawdf['comment_label'].fillna(1.0).astype(int)
rawdf = linear_join(rawdf,datadf)

# pdb.set_trace(header='post join')

rawdf[['comment_text','comment_label']].to_csv(args.xpath)
zcount = len(rawdf[rawdf['comment_label']==0])
zprop = zcount / len(rawdf)
print(f"class distribution: 0:({zprop}|{zcount}) 1:({1-zprop}|{len(rawdf)-zcount})")

# pdb.set_trace(header='last trace')

clear_cache()