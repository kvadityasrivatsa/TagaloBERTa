# TagaloBERTa
Tagalog-English code-mixed hatespeech detection

## Getting Started

### Installation
```
$ git clone https://github.com/kvadityasrivatsa/TagaloBERTa.git
$ cd TagaloBERTa/
$ pip install -r requirements.txt
$ mkdir ./data
```
### Generating Labels
Use the following command to label (0/1) a csv file for hate-speech.
```
$ python3 generate_labels.py
```
Make sure the csv contains the following column(s):
- "comment_text": One sample text per row for labelling
- "annotation": Binary label per sample text. (Only for generating metric scores)
