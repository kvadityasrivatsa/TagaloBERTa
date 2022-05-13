import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RE_PATTERNS = {'urls':r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",
               'not_alpha_num':r"[^a-zA-Z0-9']",
               }
RE_PATTERNS = {k:re.compile(v) for k,v in RE_PATTERNS.items()}

def clean_label_df(df,max_seq_len=512):

    texts = df['comment_text']
    texts = [re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)",'',s) for s in texts]
    texts = [re.sub(r"[^a-zA-Z0-9']",' ',s) for s in texts]
    texts = [re.sub(r"[ ]+",' ',s) for s in texts]
    texts = [str(s)[:512] for s in texts]
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
    datadf['comment_text'] = [str(s)[:512].strip() for s in datadf['comment_text']]
    datadf = datadf[[len(l)>5 for l in datadf['comment_text']]]

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
        traindf = traindf.reset_index(drop=True)
        testdf = testdf.reset_index(drop=True)
        traindf.to_csv('./data/train.csv',index=False)
        testdf.to_csv('./data/test.csv',index=False)
        dataset = load_dataset('csv', data_files={'train':'./data/train.csv',
                                                  'test':'./data/test.csv'})

    else:
        datadf = datadf.reset_index(drop=True)
        datadf.to_csv('./data/data.csv',index=False)
        dataset = load_dataset('csv',data_files={'eval':'./data/data.csv'})

    return dataset, datadf