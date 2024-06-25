import torch
import numpy as np
import random
import argparse
import glob
import os, sys
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from tqdm import tqdm
from sklearn import metrics
from torch import nn
import datetime
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from torch.utils.tensorboard import SummaryWriter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from accelerate import Accelerator
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_polynomial_decay_schedule_with_warmup
)
import pdb
from constants import SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, MODEL_INPUTS, PADDED_INPUTS, PHQ_TOKENS
from load_data import get_dataset
from utils import call_questions, calculate_similarity
import pickle
import gzip
from make_logger import CreateLogger

import torch.distributed as dist
import torch.multiprocessing as mp

import warnings


class DialT5(nn.Module):
    def __init__(self, args, logger):
        super(DialT5, self).__init__()
        self.args = args
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        num_new_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        logger.info("***** Load model *****")
        self.set_seed(self.args.seed)
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.model.resize_token_embeddings(self.args.vocab_size)
        self.writer = SummaryWriter(log_dir=f'{self.args.output_dir}/tensorboard')
        self.args.max_len = min(self.args.max_len, self.model.config.n_positions)  # No generation bigger than model size
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token, self.pos_token, self.neg_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id, self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        if self.accelerator.is_local_main_process:
            logger.info('***** Loading the optimizer ******')
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.best_loss = sys.float_info.max
        self.last_epoch = 0
        if self.accelerator.is_local_main_process:
            logger.info('***** Loading train & valid dataset *****')
        if self.args.mode == 'train':
            if os.path.isfile(f"{self.args.data_dir}/train_ids2.pickle"):
                with gzip.open(f"{self.args.data_dir}/train_ids2.pickle", 'rb') as f:
                    train_set = pickle.load(f)
            else:
                train_set = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.args)
                if self.accelerator.is_local_main_process:
                    logger.info("***** Saving train_ids *****")
                    with gzip.open(f"{self.args.data_dir}/train_ids2.pickle", 'wb') as f:
                        pickle.dump(train_set, f)
            if os.path.isfile(f"{self.args.data_dir}/valid_ids2.pickle"):
                with gzip.open(f"{self.args.data_dir}/valid_ids2.pickle", 'rb') as f:
                    valid_set = pickle.load(f)
            else:
                valid_set = get_dataset(tokenizer=self.tokenizer, type_path="valid", args=self.args)
                if self.accelerator.is_local_main_process:
                    logger.info("***** Saving valid_ids *****")
                    with gzip.open(f"{self.args.data_dir}/valid_ids2.pickle", 'wb') as f:
                        pickle.dump(valid_set, f)
            
            

            self.train_loader = DataLoader(train_set, batch_size=self.args.train_batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.args.eval_batch_size,  shuffle=False)
            
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
            
            num_batches = len(self.train_loader)
            args.total_train_steps = num_batches * self.args.num_train_epochs
            args.warmup_steps = int(args.total_train_steps * args.warmup_ratio)

            self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_train_steps, power=2)
            
        if self.args.ckpt_name is not None:
            self.args.model_ckpt_path = f"{self.args.output_dir}/T5models/{self.args.ckpt_name}"
            ckpt_path = f"{self.args.output_dir}/others/{self.args.ckpt_name}.ckpt"
            if os.path.isdir(self.args.model_ckpt_path):
                self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_ckpt_path)
                self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
                self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_train_steps, power=2)
                if self.accelerator.is_local_main_process:
                    logger.info("***** Load model from the specific checkpoint *****")
                ckpt = torch.load(ckpt_path, map_location=self.device)

                if self.args.mode == 'train':
                    if self.accelerator.is_local_main_process:
                        logger.info("The training restarts with the specified checkpoint")
                    self.best_loss = ckpt['best_loss']
                    self.last_epoch = ckpt['epoch']
                else:
                    if self.accelerator.is_local_main_process:
                        logger.info("The inference will start with the specified checkpoint")
            else:
                if self.accelerator.is_local_main_process:
                    logger.info("The specified checkpoint does not exist")
                if self.args.mode == 'train':
                    if self.accelerator.is_local_main_process:
                        logger.info("The training will start from scratch")
                else:
                    if self.accelerator.is_local_main_process:
                        logger.info("Cannot inference")
                    exit()

        self.model.to(self.device)
        self.model, self.optim, self.train_loader, self.valid_loader, self.scheduler = self.accelerator.prepare(self.model, self.optim, self.train_loader, self.valid_loader, self.scheduler)
        if self.accelerator.is_local_main_process:
            logger.info("***** setting finished *****")
            
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_embedding(self, text):
        ids = self.tokenizer.encode_plus([text], return_tensors = 'pt', padding = 'max_length')
        try:
            output = self.model.encoder(ids['input_ids'].to('cuda'), attention_mask=ids['attention_mask'].to('cuda')).last_hidden_state.mean(dim=-1)
        except:
            output = self.model.module.encoder(ids['input_ids'].to('cuda'), attention_mask=ids['attention_mask'].to('cuda')).last_hidden_state.mean(dim=-1)
        return output.detach().cpu().numpy()
        
    def running_train(self):
        self.set_seed(self.args.seed)
        if self.accelerator.is_local_main_process:
            logger.info("***** Running training *****")
        is_main = self.accelerator.is_local_main_process
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.args.num_train_epochs + start_epoch):
            self.model.train()
            txt = f"#"*50 + f"Epoch: {epoch}" + "#"*50
            if self.accelerator.is_local_main_process:
                logger.info(txt)
            train_losses = []
            train_ppls = []
            
            for i, batch in enumerate(tqdm(self.train_loader, disable = not is_main, desc = 'Training..')):
                with self.accelerator.accumulate(self.model):
                    trues = []
                    preds = []
                    input_ids, source_mask, labels, target_mask = batch
                    input_ids, source_mask, labels, target_mask = input_ids.to(self.device), source_mask.to(self.device), labels.to(self.device), target_mask.to(self.device)
                    flag = [False]*len(labels)

                    for j, lab in enumerate(labels):
                        lab = lab.tolist()
                        trues.append(lab[1])
                        if self.neg_id in lab:
                            flag[j] = True
                        else:
                            flag[j] = False
                    
                    for i in range(len(flag)):
                        if flag[i]:
                            is_ids = False
                            best_sim = -100
                            questions = call_questions()
                            try:
                                label_embd = self.model.encoder(labels[i].unsqueeze(0), attention_mask=target_mask[i].unsqueeze(0)).last_hidden_state.mean(dim=-1).detach().cpu().numpy()
                                input_embd = self.model.encoder(input_ids[i].unsqueeze(0), attention_mask=source_mask[i].unsqueeze(0)).last_hidden_state.mean(dim=-1).detach().cpu().numpy()
                            except:
                                label_embd = self.model.module.encoder(labels[i].unsqueeze(0), attention_mask=target_mask[i].unsqueeze(0)).last_hidden_state.mean(dim=-1).detach().cpu().numpy()
                                input_embd = self.model.module.encoder(input_ids[i].unsqueeze(0), attention_mask=source_mask[i].unsqueeze(0)).last_hidden_state.mean(dim=-1).detach().cpu().numpy()
                            sim = calculate_similarity(label_embd, input_embd)
                            if sim > best_sim:
                                best_sim = sim
                                is_ids = True
                            for key in questions.keys():
                                text = random.choice(questions[key])
                                emb_text = self.get_embedding(text)
                                sim= calculate_similarity(emb_text, input_embd)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_q = text
                                    is_ids = False
                            if is_ids:
                                pass
                            else:
                                new_label = self.emo_token + ' ' + self.tokenizer.decode(labels[i][1]) + self.sys_token + ' ' + best_q
                                new_label = self.tokenizer.tokenize(new_label)
                                new_label = self.tokenizer.encode_plus(new_label, return_tensors = 'pt', padding = 'max_length')
                                labels[i] = new_label['input_ids']
                                target_mask[i] = new_label['attention_mask']
                    
                    outputs = self(input_ids=input_ids, attention_mask=source_mask, labels=labels)
                    loss, logits = outputs[0], outputs[1]
                    self.optim.zero_grad()
                    self.accelerator.backward(loss)
                    self.optim.step()
                    self.scheduler.step()

                train_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                train_ppls.append(ppl)

                del loss, logits, outputs
            
            train_losses = [loss.item() for loss in train_losses]
            train_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
            train_loss = np.mean(train_losses)
            train_ppl = np.mean(train_ppls)
            if is_main:
                logger.info(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")


            self.last_epoch += 1
            
            valid_loss, valid_ppl, bleu1, mean2, mean3, mean4, acc, f1 = self.validation()
            self.accelerator.wait_for_everyone()
            if is_main:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    unwrapped_model.save_pretrained(f"{self.args.output_dir}/T5models/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}", save_function=self.accelerator.save)
                    logger.info("*"*10 + "Current best checkpoint is saved." + "*"*10)
                    state_dict = {
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'epoch': self.last_epoch
                }
                    torch.save(state_dict, f"{self.args.output_dir}/others/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")

                logger.info(f"Best valid loss: {self.best_loss}")
                logger.info(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")
                logger.info(f"BLEU1: {bleu1} || BLEU2: {mean2} || BLEU3: {mean3} || BLEU4: {mean4}")
                logger.info(f"Accuracy: {acc} || F1: {f1}")

            
        if is_main:  
            logger.info("Training finished!")

    def validation(self):
        is_main = self.accelerator.is_local_main_process
        if is_main:
            logger.info("***** Running validation *****")

        self.model.eval()

        valid_losses = []
        valid_ppls = []
        valid_bleu1 = []
        valid_mean2 = []
        valid_mean3 = []
        valid_mean4 = []
        valid_acc = []
        valid_f1 = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader, disable = not is_main, desc = 'Validating..')):
                input_ids, source_mask, labels, target_mask = batch
                input_ids, source_mask, labels, target_mask = input_ids.to(self.device), source_mask.to(self.device), labels.to(self.device), target_mask.to(self.device)
                trues = []
                preds = []
                for lab in labels:
                    lab = lab.tolist()
                    if self.neg_id in lab:
                        trues.append(self.neg_id)
                    elif self.pos_id in lab:
                        trues.append(self.pos_id)
                try:          
                    generated = self.model.module.generate(input_ids=input_ids, attention_mask=source_mask, max_length=self.args.max_output_length)
                except:
                    generated = self.model.generate(input_ids=input_ids, attention_mask=source_mask, max_length=self.args.max_output_length)

                for i, out in enumerate(generated):
                    out = out.tolist()
                    if self.neg_id in out:
                        preds.append(self.neg_id)
                    elif self.pos_id in out:
                        preds.append(self.pos_id)
                    else:
                        preds.append(-1)

                outputs = self(input_ids=input_ids, attention_mask=source_mask, labels=labels)
                
                loss, logits = outputs[0], outputs[1]
                reference = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
                candidates = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in generated]

                for ref, cand in zip(reference, candidates):
                    import pdb; pdb.set_trace()
                    valid_bleu1.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7,weights=(1, 0, 0, 0)))
                    valid_mean2.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7, weights=(0.5, 0.5, 0, 0)))
                    valid_mean3.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7, weights=(0.33, 0.33, 0.33, 0)))
                    valid_mean4.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7, weights=(0.25, 0.25, 0.25, 0.25)))
                valid_losses.append(loss.detach())
                valid_ppls.append(torch.exp(loss.detach()))
                valid_acc.append(metrics.accuracy_score(trues, preds))
                valid_f1.append(metrics.f1_score(trues, preds, average='macro'))
        valid_losses = [loss.item() for loss in valid_losses]
        valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]

        valid_loss = np.mean(valid_losses)
        valid_ppl = np.mean(valid_ppls)
        bleu1 = np.mean(valid_bleu1)
        mean2 = np.mean(valid_mean2)
        mean3 = np.mean(valid_mean3)
        mean4 = np.mean(valid_mean4)
        valid_acc = np.mean(valid_acc)
        valid_f1 = np.mean(valid_f1)
        if math.isnan(valid_ppl):
            valid_ppl = 1e+8
        return valid_loss, valid_ppl, bleu1, mean2, mean3, mean4, valid_acc, valid_f1

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(self.args.seed) # if use multi-GPU

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset/')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--log_name', type=str, default='wIwC')
    parser.add_argument('--model_name_or_path', type=str, default='t5-small')
    parser.add_argument('--max_turns', type=int, default=7)
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--mode', type=str, default='valid')
    parser.add_argument('--usr_token', type=str, default='<usr>')
    parser.add_argument('--sys_token', type=str, default='<sys>')
    parser.add_argument('--bos_token', type=str, default='<s>')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")
    # parser.add_argument('--ckpt_name', type=str, default='best_ckpt_epoch=5_valid_loss=0.0233', help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default = "quit!")
    parser.add_argument('--special_eos', type=str, default = "</s>")
    parser.add_argument('--pad_token', type=str, default = '<pad>')

    args = parser.parse_args()
    date = datetime.datetime.now()
    date_txt = date.strftime("%Y%m%d%A%H:%M:%S")
    logger = CreateLogger("DialwIwC", f"{args.output_dir}/logs/{args.log_name}{date_txt}.log")
    NGPU = torch.cuda.device_count()
    args.n_gpus = NGPU

    args.local_rank = [i for i in range(args.n_gpus)]
    args.num_workers = NGPU
    args.train_batch_size = int(args.train_batch_size / args.n_gpus)
    args.eval_batch_size = int(args.eval_batch_size / args.n_gpus)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps / args.n_gpus)
    args.lr = args.lr * args.n_gpus

    if args.n_gpus > 1:
        mp.set_start_method('spawn')

    model = DialT5(args, logger)
    model.validation()
