import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from underthesea import word_tokenize

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import datasets
device = "cuda" if torch.cuda.is_available() else "cpu"


import re

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class config:
    train_path = 'data/train.csv'
    model_name = 'vinai/phobert-base-v2'
    max_length = 256
    batch_size = 64
    num_workers = 32
    
df = pd.read_csv(config.train_path)

df_train, df_val = train_test_split(df, test_size=0.2, random_state=84)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
train_dataset = datasets.Dataset.from_pandas(df_train)
val_dataset = datasets.Dataset.from_pandas(df_val)
label2id = {'SUPPORTED': 0, 'REFUTED': 1,'NEI': 2}
id2label = {0: 'SUPPORTED', 1:'REFUTED', 2:'NEI'}
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=3, id2label=id2label, label2id=label2id).to(device)
def PreprocessDataset(examples):    
    inputs =  tokenizer(
            text=examples['top_bm25'],
            text_pair=examples['claim_tokenizer'],
            max_length=config.max_length,
            padding='max_length',
            truncation='only_first',
            return_tensors='pt',
            
        )
    labels = examples['verdict_label']
    inputs.update({'labels': labels})
    return inputs
train_datasets = train_dataset.map(PreprocessDataset, batched=True, batch_size=64,remove_columns=train_dataset.column_names)
valid_datasets = val_dataset.map(PreprocessDataset, batched=True,  batch_size=64,remove_columns=train_dataset.column_names)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='micro')  # Có thể thay 'weighted' bằng 'micro', 'macro', hoặc None tùy vào yêu cầu của bạn
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

training_args = TrainingArguments(
    output_dir='models/model_v5',          # output directory
    num_train_epochs=10,              # total number of training epochs
    learning_rate=1e-5,              # learning rate
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    # gradient_accumulation_steps=2,   # Number of updates steps to accumulate before performing a backward/update pass.
    warmup_steps=250,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=250,
    eval_steps=250,
    evaluation_strategy='steps',
    load_best_model_at_end=True,
    metric_for_best_model='f1_score',
    greater_is_better=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_datasets,
    eval_dataset=valid_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('finals/model_v5')