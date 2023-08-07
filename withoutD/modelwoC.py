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

from make_logger import CreateLogger

import torch.distributed as dist
import torch.multiprocessing as mp

import warnings
# warnings.filterwarnings("ignore")


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
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(['pos', 'neg'])
        logger.info('***** Loading the optimizer ******')
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.best_loss = sys.float_info.max
        self.last_epoch = 0

        logger.info('***** Loading train & valid dataset *****')
        if self.args.mode == 'train':
            train_set = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.args)
            valid_set = get_dataset(tokenizer=self.tokenizer, type_path="valid", args=self.args)

            self.train_loader = DataLoader(train_set, batch_size=self.args.train_batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.args.eval_batch_size,  shuffle=False)
            
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
            
            num_batches = len(self.train_loader)
            args.total_train_steps = num_batches * self.args.num_train_epochs
            args.warmup_steps = int(args.total_train_steps * args.warmup_ratio)

            self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_train_steps, power=2)
            
            self.model.to(self.device)
            self.model, self.optim, self.train_loader, self.valid_loader = self.accelerator.prepare(self.model, self.optim, self.train_loader, self.valid_loader)
    


        if self.args.ckpt_name is not None:
            ckpt_path = f"{self.args.output_dir}/T5models/{self.args.ckpt_name}.ckpt"
            if os.path.exists(ckpt_path):
                logger.info('***** Loading checkpoint *****')
                ckpt = torch.load(ckpt_path, map_location=self.device)
                pre_trained = ckpt['model_state_dict']
                new_model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pre_trained.items() if k in new_model_dict.keys()}
                if len(pretrained_dict) == 0:
                    pretrained_dict = {k.replace('module.',''): v for k, v in pre_trained.items() if k.replace('module.','') in new_model_dict.keys()}
                if len(pretrained_dict) == 0:
                    logger.info("Cannot load the checkpoint")
                
                new_model_dict.update(pretrained_dict)
                self.model.load_state_dict(new_model_dict)
                del pre_trained, new_model_dict
                torch.cuda.empty_cache()

                if self.args.mode == 'train':
                    logger.info("The training restarts with the specified checkpoint")
                    # self.optim.load_state_dict(ckpt['optimizer_state_dict'])
                    # self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    self.best_loss = sys.float_info.max
                    self.last_epoch = 0
                else:
                    logger.info("The inference will start with the specified checkpoint")
            else:
                logger.info("The specified checkpoint does not exist")
                if self.args.mode == 'train':
                    logger.info("The training will start from scratch")
                else:
                    logger.info("Cannot inference")
                    exit()
        
        logger.info("***** setting finished *****")
            
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_embedding(self, dicts):        
        texts = []
        for key in dicts.keys():
            texts.append(dicts[key])
        ids = self.tokenizer.encode_plus(texts, return_tensors = 'pt', padding = 'max_length').cuda()
        output = self.model.encoder(ids['input_ids'], attention_mask=ids['attention_mask']).hidden_states[-1].mean(dim=-1)
        return output
        
    def running_train(self):
        self.set_seed(self.args.seed)
        logger.info("***** Running training *****")

        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.args.num_train_epochs + start_epoch):
            self.model.train()
            txt = f"#"*50 + f"Epoch: {epoch}" + "#"*50
            logger.info(txt)
            train_losses = []
            train_ppls = []
            
            for i, batch in enumerate(tqdm(self.train_loader)):

                input_ids, source_mask, labels, target_mask = batch
                input_ids, source_mask, labels, target_mask = input_ids.to(self.device), source_mask.to(self.device), labels.to(self.device), target_mask.to(self.device)
                outputs = self(input_ids=input_ids, attention_mask=source_mask, labels=labels)
                loss, logits = outputs[0], outputs[1]

                self.optim.zero_grad()
                # loss.backward()
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
            logger.info(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Perplexity/train', train_ppl, epoch)

            self.last_epoch += 1
            
            valid_loss, valid_ppl, bleu1, mean2, mean3, mean4 = self.validation()

            if valid_loss < self.best_loss :
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'epoch': self.last_epoch
                }

                torch.save(state_dict, f"{self.args.output_dir}/T5models/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
                logger.info("*"*10 + "Current best checkpoint is saved." + "*"*10)
                logger.info(f"{self.args.output_dir}/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
            elif epoch % 5 == 0:
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'epoch': self.last_epoch
                }
                torch.save(state_dict, f"{self.args.output_dir}/T5models/ckpt_epoch={epoch}_valid_loss={round(valid_loss, 4)}.ckpt")
                logger.info(f"{self.args.output_dir}/ckpt_epoch={epoch}_valid_loss={round(valid_loss, 4)}.ckpt")

            logger.info(f"Best valid loss: {self.best_loss}")
            logger.info(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")
            logger.info(f"BLEU1: {bleu1} || BLEU2: {mean2} || BLEU3: {mean3} || BLEU4: {mean4}")

            
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            
            self.writer.add_scalars("Losses", {
                'train': train_loss, 
                'valid': valid_loss,
            }, epoch)
            self.writer.add_scalars("PPLs", {
                'train': train_ppl,
                'valid': valid_ppl,
            }, epoch)

            self.writer.add_scalars("BLEUs", {
                'bleu1': bleu1,
                'bleu2': mean2,
                'bleu3': mean3,
                'bleu4': mean4,
            }, epoch)
              
        logger.info("Training finished!")

    def validation(self):
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
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, source_mask, labels, target_mask = batch
                input_ids, source_mask, labels, target_mask = input_ids.to(self.device), source_mask.to(self.device), labels.to(self.device), target_mask.to(self.device)

                try:          
                    generated = self.model.module.generate(input_ids=input_ids, attention_mask=source_mask, max_length=self.args.max_output_length)
                except:
                    generated = self.model.generate(input_ids=input_ids, attention_mask=source_mask, max_length=self.args.max_output_length)

                outputs = self(input_ids=input_ids, attention_mask=source_mask, labels=labels)
                
                loss, logits = outputs[0], outputs[1]
                reference = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
                candidates = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in generated]

                for ref, cand in zip(reference, candidates):
                    valid_bleu1.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7,weights=(1, 0, 0, 0)))
                    valid_mean2.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7, weights=(0.5, 0.5, 0, 0)))
                    valid_mean3.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7, weights=(0.33, 0.33, 0.33, 0)))
                    valid_mean4.append(bleu_score.sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method7, weights=(0.25, 0.25, 0.25, 0.25)))
                valid_losses.append(loss.detach())
                valid_ppls.append(torch.exp(loss.detach()))

        valid_losses = [loss.item() for loss in valid_losses]
        valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]

        valid_loss = np.mean(valid_losses)
        valid_ppl = np.mean(valid_ppls)
        bleu1 = np.mean(valid_bleu1)
        mean2 = np.mean(valid_mean2)
        mean3 = np.mean(valid_mean3)
        mean4 = np.mean(valid_mean4)

        if math.isnan(valid_ppl):
            valid_ppl = 1e+8
        return valid_loss, valid_ppl, bleu1, mean2, mean3, mean4

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
    parser.add_argument('--output_dir', type=str, default='./outputs_woC')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_name', type=str, default='DialT5')
    parser.add_argument('--model_name_or_path', type=str, default='t5-small')
    parser.add_argument('--max_turns', type=int, default=7)
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--usr_token', type=str, default='<usr>')
    parser.add_argument('--sys_token', type=str, default='<sys>')
    parser.add_argument('--bos_token', type=str, default='<s>')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")
    # parser.add_argument('--ckpt_name', type=str, default='best_ckpt_epoch=10_valid_loss=0.0123', help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default = "quit!")
    parser.add_argument('--special_eos', type=str, default = "</s>")
    parser.add_argument('--pad_token', type=str, default = '<pad>')

    args = parser.parse_args()
    date = datetime.datetime.now()
    date_txt = date.strftime("%Y%m%d%A%H:%M:%S")
    logger = CreateLogger("DialT5", f"{args.output_dir}/logs/{args.log_name}{date_txt}.log")
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
    model.running_train()
