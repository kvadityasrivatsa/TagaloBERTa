import warnings
warnings.filterwarnings('ignore')

import os
import shutil
from time import time
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


mtitle ="TagaloBERTa"
mpath = "./TagaloBERTa"
dpath = "clean_text_30M.txt" 
batch_size = 16

if os.path.exists(mpath):
	shutil.rmtree(mpath)
os.makedirs(mpath)

start = time()

paths = [dpath]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Training Tokenizer
now = time(); elapsed = now - start; print(f"training tokenizer | elapsed: {elapsed} s")
tokenizer.train(files=paths, vocab_size=50_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(mtitle)

now = time(); elapsed = now - start; print(f" | elapsed: {elapsed} s")
tokenizer = ByteLevelBPETokenizer(
    os.path.join(mpath,"vocab.json"),
    os.path.join(mpath,"merges.txt"),
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

now = time(); elapsed = now - start; print(f"setting up RoBERTa config | elapsed: {elapsed} s")
config = RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
	cache_dir=os.path.join(mpath,'cache')
)

tokenizer = RobertaTokenizer.from_pretrained(mpath, max_len=512)
model = RobertaForMaskedLM(config=config)

print(f"Model parameters: {model.num_parameters()}")

now = time(); elapsed = now - start; print(f"initializing dataset | elapsed: {elapsed} s")
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=dpath,
    block_size=batch_size,
)

now = time(); elapsed = now - start; print(f"building data collator | elapsed: {elapsed} s")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

now = time(); elapsed = now - start; print(f"initializing training args | elapsed: {elapsed} s")
training_args = TrainingArguments(
    output_dir=mpath,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_gpu_train_batch_size=batch_size,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

now = time(); elapsed = now - start; print(f"training model | elapsed: {elapsed} s")
trainer.train()

now = time(); elapsed = now - start; print(f"saving model | elapsed: {elapsed} s")
trainer.save_model(mpath)