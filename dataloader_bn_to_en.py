import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tokenizer as tk
from dataset import BilingualDataset, causal_mask
from sampletransformers import build_transformer


def get_ds(config):
    ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}")

    tokenizer_src = tk.get_or_build_tokenizer(config, ds, config['lang_src'])
    tokenizer_tgt = tk.get_or_build_tokenizer(config, ds, config['lang_tgt'])

    train_data = ds["train"] 
    val_data = ds["validation"]    
    test_data = ds["test"]
    
    train_ds = BilingualDataset(train_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
        
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    train_dataloader= DataLoader(train_ds,batch_size=config['batch_size'], shuffle=True)
    val_dataloader= DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model=build_transformer(vocab_src_len,vocab_tgt_len, config['seq_len'], config['seq_len',config['d_model']])
    return model





