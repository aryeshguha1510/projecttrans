import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import src.tokenizer as tk
from dataset import BilingualDataset, causal_mask
from src.sampletransformers import build_transformer
import src.training as training
import random
from src import training


def get_ds(config):

    ds_raw_train = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split="train[0:10000]")
    #ds_raw_train = load_dataset("Helsinki-NLP/opus-100", "bn-en", split="train[0:10000]")
    #ds_raw_train = ds_raw['train'].shuffle(seed=42).select([i for i in range(10000)])
    ds_raw_val = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split="validation")
    #ds_raw_val = load_dataset("Helsinki-NLP/opus-100", "bn-en", split="validation")
    #ds_raw_val = ds_raw['validation']
    
    # Build tokenizers
    tokenizer_src = training.get_or_build_tokenizer(config, ds_raw_train, config['lang_src'])
    tokenizer_tgt = training.get_or_build_tokenizer(config, ds_raw_train, config['lang_tgt'])
 
    train_ds_size = int(len(ds_raw_train))
    val_ds_size = int(len(ds_raw_val))
    print(train_ds_size)
    print(val_ds_size)

    train_ds = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_raw_val, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw_train:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model




