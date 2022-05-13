# TagaloBERTa
Tagalog-English code-mixed hatespeech detection

## Getting Started

### Installation
```
$ git clone https://github.com/kvadityasrivatsa/TagaloBERTa.git
$ cd TagaloBERTa/
$ pip install -r requirements.txt
```
### Generating Labels
Use the following command to label (0/1) a csv file for hate-speech.
```
$ python3 generate_labels.py --raw-data <path to input csv> --output-path <path to output csv>
```
Make sure the csv contains the following column(s):
- "comment_id": Unique ID per row (output may be a subset of these ids).
- "comment_text": One sample text per row for labelling.
- "comment_label": (Optional) Binary label per sample text. (Only for generating metric scores).


