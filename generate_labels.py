import os
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from dataloader import fetch_base_encoder, fetch_finetuned_model, load_huggingface_data, load_custom_data
from preprocessing import preprocess_label_data

argp = argparse.ArgumentParser()
argp.add_argument('--raw-data',type=str,dest='rpath',required=True,default=None)
argp.add_argument('--tuned-model',type=str,dest='tpath',default='TagaloBERTa_RoBERTa_Bi10_30M.model')
argp.add_argument('--base-model',type=str,dest='bpath',default='TagaloBERTa_30M')
argp.add_argument('--tuned-model-split',type=str,dest='tsplit',default='test')
args = argp.parse_args()

print('loading data to be labelled.')
datadf = load_huggingface_data(args.rpath,args.tsplit)
if not datadf:
    datadf = load_custom_data(args.rpath)

print('preprocessing data.')
dataset = preprocess_label_data(datadf,split=False)

print('tokenizing data.')
tokenizer = AutoTokenizer.from_pretrained(args.bpath)
def tokenize_function(examples):
    return tokenizer(examples["text"], 
                     padding="max_length", 
                     truncation=True,
                     max_length=512)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print('loading model.')
model_args = TrainingArguments()
model_path = fetch_finetuned_model(args.tpath)
model = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=2),
    args=model_args,
)

print('generating labels.')
_pred = model.predict(dataset)
pred = np.argmax(_pred.prediction,axis=1)
if 'labels' in dataset:
    print(classification_report(pred,dataset['labels']))

