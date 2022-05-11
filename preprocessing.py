import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def clean_label_df(df):
    texts, labels = df['text'], df['labels']
        
    texts = [re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",'',s) for s in texts]
    texts = [re.sub(r"[^a-zA-Z0-9']",' ',s) for s in texts]
    texts = [re.sub(r"[ ]+",' ',s) for s in texts]
    texts = [str(s)[:512] for s in texts]

    labels = [1.0 if int(l)==1 else 0.0 for l in labels]
    labels = [1 if s else 0 for s in labels]

    df['text'], df['labels'] = texts, labels
    return df

def preprocess_label_data(datadf,split=False,test_size=None):
    # datadf = datadf.dropna()
    datadf[['text']].fillna('',inplace=True)
    datadf = clean_label_df(datadf)
    datadf['text'] = [str(s)[:512].strip() for s in datadf['text']]
    if 'labels' in datadf.columns:
        datadf[['labels']].fillna(0,inplace=True)
        datadf['labels'] = [1 if int(l)==1 else 0 for l in datadf['labels']]

    if split:
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