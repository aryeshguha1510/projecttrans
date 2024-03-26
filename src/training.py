from sampletransformers import build_transformer
from  DSfn import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import wandb
from nltk.translate.bleu_score import sentence_bleu


import argparse
parser = argparse.ArgumentParser(description='Input hyperparameters')
parser.add_argument('--k',metavar='API Key', type=int, help='Enter the API Key')
parser.add_argument('batch_size',metavar='Batch Size', type=int, help='Enter the batch size')
parser.add_argument('num_epochs',metavar='Number of Epochs', type=int, help='Enter the number of epochs')
parser.add_argument('seq_len',metavar='Sequence Length', type=int, help='Enter the Sequence Length')
parser.add_argument('d_model',metavar='d_model', type=int, help='d_model')
parser.add_argument('lr',metavar='Learning Rate', type=float, help='Enter the learning rate')

args = parser.parse_args()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
   
    ds_raw_train = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split="train[0:10000]")
    #ds_raw_train = load_dataset("Helsinki-NLP/opus-100", "bn-en", split="train[0:10000]")
    #ds_raw_train = ds_raw['train'].shuffle(seed=42).select([i for i in range(10000)])
    ds_raw_val = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split="validation")
    #ds_raw_val = load_dataset("Helsinki-NLP/opus-100", "bn-en", split="validation")
    #ds_raw_val = ds_raw['validation']
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw_train, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw_train, config['lang_tgt'])
 
    train_ds_size = int(len(ds_raw_train))
    val_ds_size = int(len(ds_raw_val))
    print(train_ds_size)
    print(val_ds_size)

    train_ds = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], args.seq_len)
    val_ds = BilingualDataset(ds_raw_val, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], args.seq_len)

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
    

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, args.seq_len, args.seq_len, d_model=args.d_model)
    return model

def train_model(config):
    # Define the device
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda:0'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, args.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch_idx, batch in enumerate(batch_iterator):

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            _, predicted_ids = torch.max(proj_output, dim=2)            
            #logging loss
            wandb.log({"train_loss": loss.item(), "global_step": global_step})
            if batch_idx % len(train_dataloader) == 0: 
                source_tokens = [tokenizer_src.id_to_token(tok_id) for tok_id in encoder_input[0]]
                predicted_tokens = [tokenizer_tgt.id_to_token(tok_id) for tok_id in predicted_ids[0]]
                source_sentence = ' '.join(source_tokens).split('[PAD]')[0]  
                
            
                predicted_sentence = ' '.join(predicted_tokens).split('[PAD]')[0]  
                #target sentence
                target_tokens = [tokenizer_tgt.id_to_token(tok_id) for tok_id in label[0]]
                target_sentence = ' '.join(target_tokens).split('[PAD]')[0]
                print(f"Actual Target: {target_sentence}")

                print(f"Source Text: {source_sentence}\n")
                print(f"Sample Prediction: {predicted_sentence}\n")

                reference = [target_sentence.split()]
                candidate = predicted_sentence.split()
                score = sentence_bleu(reference, candidate)
                print(f"BLEU Score: {score}")
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, args.seq_len, device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

wandb.login(args.k)
wandb.init(
    # set the wandb project where this run will be logged
    project="BengToEng",
    
    # track hyperparameters and run metadata
    config={
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,
    "lr": args.lr,
    "seq_len": args.seq_len,
    "d_model": args.d_model,
    }
)
 


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)