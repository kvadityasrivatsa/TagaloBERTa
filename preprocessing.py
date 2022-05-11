import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RE_PATTERNS = {'urls':r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",
               'not_alpha_num':r"[^a-zA-Z0-9']",
               }
RE_PATTERNS = {k:re.compile(v) for k,v in RE_PATTERNS.items()}

def clean_label_df(df,max_seq_len=512):
    texts = df['text']
    labels = df['labels'] if 'labels' in df.columns else None
    for i in range(len(texts)):
        text = texts[i]
        try:
            for p in RE_PATTERNS.values():
                text = re.sub(p,'',text)
            texts[i] = text
        except:
            texts[i] = ''
    texts = [re.sub(r"[ ]+",' ',s) for s in texts]
    texts = [str(s)[:max_seq_len] for s in texts]
    df['text'] = texts

    if 'labels' in df.columns:
        labels = [1.0 if int(l)==1 else 0.0 for l in labels]
        labels = [1 if s else 0 for s in labels]
        df['labels'] = labels

    return df

def preprocess_label_data(datadf,split=False,test_size=None):
    # datadf = datadf.dropna()
    datadf[['text']].fillna('nan',inplace=True)
    datadf = clean_label_df(datadf)
    datadf['text'] = [str(s)[:512].strip() for s in datadf['text']]
    if 'labels' in datadf.columns:
        datadf[['labels']].fillna(0,inplace=True)
        datadf['labels'] = [1 if int(l)==1 else 0 for l in datadf['labels']]

    if split:
        datadf = datadf.dropna()
        datadf = datadf[[len(l)>5 for l in datadf['text']]]
        print(f'before pruning: {len(datadf)}')
        datadf = datadf.drop_duplicates(subset=['text'],keep='first')
        print(f'after pruning: {len(datadf)}')

        traindf, testdf = train_test_split(datadf,test_size=test_size)
        traindf = traindf.reset_index(drop=True)
        testdf = testdf.reset_index(drop=True)

        traindf.to_csv('./data/train.csv',index=False)
        testdf.to_csv('./data/test.csv',index=False)

        dataset = load_dataset('csv',data_files={'train':'train.csv',
                                                 'test':'test.csv'})

    else:
        datadf = datadf.reset_index(drop=True)
        datadf.to_csv('./data/data.csv',index=False)

        dataset = load_dataset('csv',data_files={'eval':'./data/data.csv'})

    return dataset