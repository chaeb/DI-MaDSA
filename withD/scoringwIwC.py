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
from load_datawoC import get_dataset, get_scoring_dataset
from utils import call_questions, calculate_similarity, Attention, TimeDistributed, MultiHeadAttention
import utils
import pickle
import gzip
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
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token, self.pos_token, self.neg_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id, self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
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
    
    def get_embedding(self, text):
        ids = self.tokenizer.encode_plus([text], return_tensors = 'pt', padding = 'max_length')
        output = self.model.encoder(ids['input_ids'].to('cuda'), attention_mask=ids['attention_mask'].to('cuda')).last_hidden_state.mean(dim=-1)
        return output.detach().cpu().numpy()
        
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
                flag = [True]*len(labels)
                for i in range(len(flag)):
                    if flag[i]:
                        is_ids = False
                        best_sim = 0
                        questions = call_questions()
                        label_embd = self.model.encoder(labels[i].unsqueeze(0), attention_mask=target_mask[i].unsqueeze(0)).last_hidden_state.mean(dim=-1).detach().cpu().numpy()
                        input_embd = self.model.encoder(input_ids[i].unsqueeze(0), attention_mask=source_mask[i].unsqueeze(0)).last_hidden_state.mean(dim=-1).detach().cpu().numpy()
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
                            new_label = self.sys_token + ' ' + best_q
                            new_label = self.tokenizer.tokenize(new_label)
                            new_label = self.tokenizer.encode_plus(new_label, return_tensors = 'pt', padding = 'max_length')
                            labels[i] = new_label['input_ids']
                            target_mask[i] = new_label['attention_mask']
                    
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


class trainADS(nn.Module):
    def __init__(self, args, logger):
        super(trainADS, self).__init__()
        self.args = args
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        num_new_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token, self.pos_token, self.neg_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id, self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)       
        self.dialog_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.dialog_model.resize_token_embeddings(len(self.tokenizer))   
        if self.args.ckpt_name is None:
            logger.info("****The checkpoint name is not specified****")
            exit()
        else:
            ckpt_path = f"{self.args.output_dir}/T5models/{self.args.ckpt_name}.ckpt"
            if os.path.exists(ckpt_path):
                logger.info('***** Loading checkpoint *****')
                ckpt = torch.load(ckpt_path, map_location=self.device)
                pre_trained = ckpt['model_state_dict']
                new_model_dict = self.dialog_model.state_dict()
                pretrained_dict = {k: v for k, v in pre_trained.items() if k in new_model_dict.keys()}
                if len(pretrained_dict) == 0:
                    pretrained_dict = {k.replace('module.',''): v for k, v in pre_trained.items() if k.replace('module.','') in new_model_dict.keys()}
                if len(pretrained_dict) == 0:
                    logger.info("Cannot load the checkpoint")
                    exit()
                new_model_dict.update(pretrained_dict)
                self.dialog_model.load_state_dict(new_model_dict)
                del pre_trained, new_model_dict
                torch.cuda.empty_cache()
                if self.args.mode == 'train':
                    logger.info("The training restarts with the specified checkpoint")
                else:
                    logger.info("The inference will start withthe specified checkpoint")
            else:
                logger.info("The specified checkpoint does not exist")
                exit()

        if os.path.isfile(f"{self.args.data_dir}/adswIwCtrain.pickle"):
            logger.info('***** Loading Train Data from pickle *****')
            with gzip.open(f"{self.args.data_dir}/adswIwCtrain.pickle", 'rb') as f:
                train_data = pickle.load(f)
        else:
            train_data = get_scoring_dataset(self.tokenizer, self.dialog_model, "train", self.args)
            with gzip.open(f"{self.args.data_dir}/adswIwCtrain.pickle", 'wb') as f:
                pickle.dump(train_data, f)
        if os.path.isfile(f"{self.args.data_dir}/adswIwCvalid.pickle"):
            logger.info('***** Loading Valid Data from pickle *****')
            with gzip.open(f"{self.args.data_dir}/adswIwCvalid.pickle", 'rb') as f:
                valid_data = pickle.load(f)
        else:
            valid_data = get_scoring_dataset(self.tokenizer, self.dialog_model, "valid", self.args)
            with gzip.open(f"{self.args.data_dir}/adswIwCvalid.pickle", 'wb') as f:
                pickle.dump(valid_data, f)

        self.train_loader = DataLoader(train_data, batch_size=self.args.ads_train_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_data, batch_size=self.args.ads_eval_batch_size,  shuffle=False)
        
        self.optim = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        self.best_loss = sys.float_info.max
        self.best_qwk = 0
        num_batches = len(self.train_loader)
        args.ads_total_stpes = num_batches * self.args.num_train_epochs
        args.ads_warmup_steps = int(args.ads_total_stpes * args.warmup_ratio)
        self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optim, num_warmup_steps=args.ads_warmup_steps, num_training_steps=args.ads_total_stpes, power=2)
        self.model = ADS(self.dialog_model, self.device)
        self.model, self.optim, self.train_loader, self.valid_loader = self.accelerator.prepare(self.model, self.optim, self.train_loader, self.valid_loader)

    def running_train(self):
        self.set_seed(self.args.seed)
        logger.info("***** Running ADS training *****")

        for epoch in range(self.args.ads_train_epochs):
            txt = f"#"*50 + f"Epoch: {epoch}" + "#"*50
            logger.info(txt)
            train_losses = []
            qwks_dict = {}
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_loader)):
                dialog, dial_mask, phq = batch
                dialog, dial_mask, phq = dialog.to(self.device), dial_mask.to(self.device), phq.to(self.device)
                y_pred = self.model(dialog, dial_mask)
                loss = utils.masked_loss_function(y_pred.squeeze(), phq.squeeze())
                pred_dict = utils.seperate_and_rescale_for_scoring(y_pred, is_phq=True)
                true_dict = utils.seperate_and_rescale_for_scoring(phq, is_phq=True)
                qwk_dict = {key: utils.kappa(true_dict[key], pred_dict[key]) for key in pred_dict.keys()}

                self.optim.zero_grad()
                self.accelerator.backward(loss)
                self.optim.step()
                self.scheduler.step()

                for key in qwk_dict:
                    if key not in qwks_dict.keys():
                        qwks_dict[key] = []
                    qwks_dict[key].append(qwk_dict[key])
                train_losses.append(loss.detach())

                del dialog, dial_mask, phq, y_pred, loss
            train_losses = [loss.item() for loss in train_losses]
            train_loss = np.mean(train_losses)
            train_qwk = {key: np.mean(qwks_dict[key]) for key in qwks_dict.keys()}
            mean_qwk = [qwk for qwk in train_qwk.values()]
            mean_qwk = np.mean(mean_qwk)
            logger.info(f"Train loss: {train_loss} || Train QWK: {mean_qwk}")
            self.writer.add_scalar('ADSLoss/train', train_loss, epoch)
            self.writer.add_scalar('QWK/train', mean_qwk, epoch)

            for key in train_qwk.keys():
                logger.info(f"Train QWK {key}: {qwks_dict[key]}")
                self.writer.add_scalar(f'QWK/train/{key}', train_qwk[key], epoch)

            valid_loss, valid_qwk, valid_qwk_dict = self.valid_ads()
            
            if valid_qwk > self.best_qwk:
                self.best_qwk = valid_qwk

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'epoch': epoch
                }
                torch.save(state_dict, f"{self.args.output_dir}/ADS/best_ckpt_epoch={epoch}_valid_loss={round(self.best_loss, 4)}.ckpt")
            logger.info(f"Best valid loss: {self.best_loss}")
            logger.info(f"Best valid QWK: {self.best_qwk}")
            logger.info(f"Valid loss: {valid_loss} || Valid QWK: {valid_qwk}")
            for key in valid_qwk_dict.keys():
                logger.info(f"Valid QWK {key}: {valid_qwk_dict[key]}")
                self.writer.add_scalar(f'QWK/valid/{key}', valid_qwk_dict[key], epoch)
            self.writer.add_scalar("ADSLoss/valid", valid_loss, epoch)
            self.writer.add_scalar("QWK/valid", valid_qwk, epoch)
        logger.info("ADS training finished!")

    def valid_ads(self):
        logger.info("***** Running ADS validation *****")
        self.model.eval()
        valid_losses = []
        qwks_dict = {}
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                dialog, dial_mask, phq = batch
                dialog, dial_mask, phq = dialog.to(self.device), dial_mask.to(self.device), phq.to(self.device)
                y_pred = self.model(dialog, dial_mask)
                loss = utils.masked_loss_function(y_pred.squeeze(), phq.squeeze())
                pred_dict = utils.seperate_and_rescale_for_scoring(y_pred, is_phq=True)
                true_dict = utils.seperate_and_rescale_for_scoring(phq, is_phq=True)
                qwk_dict = {key: utils.kappa(true_dict[key], pred_dict[key]) for key in pred_dict.keys()}

                for key in qwk_dict:
                    if key not in qwks_dict.keys():
                        qwks_dict[key] = []
                    qwks_dict[key].append(qwk_dict[key])
                valid_losses.append(loss.detach())
                del dialog, dial_mask, phq, y_pred, loss

        valid_qwk_dict = {key: np.mean(qwks_dict[key]) for key in qwks_dict.keys()}
        valid_losses = [loss.item() for loss in valid_losses]
        valid_loss = np.mean(valid_losses)
        mean_qwk = [qwk for qwk in valid_qwk_dict.values()]
        mean_qwk = np.mean(mean_qwk)
        return valid_loss, mean_qwk, valid_qwk_dict

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(self.args.seed) # if use multi-GPU

class ADS(nn.Module):
    def __init__(self, dialog_model, device, output_dim = 8):
        super(ADS, self).__init__()
        self.max_dialog_turns = 30
        self.dropout_prob = 0.3
        self.cnn_filters = 100
        self.cnn_kernerl_size = 5
        self.max_len = 512
        self.embed_dim = 512
        self.lstm_units = 100
        self.dialog_model = dialog_model
        self.device = device
        self.zcnn = TimeDistributed(nn.Conv1d(in_channels = 1024, out_channels = self.cnn_filters, kernel_size=self.cnn_kernerl_size, padding=0))
        self.avg_zcnn = TimeDistributed(Attention())
        self.pos_hz_MA_list = nn.ModuleList([MultiHeadAttention(self.embed_dim, 100, 2) for _ in range(output_dim)])
        self.pos_hz_MA_lstm_list = nn.ModuleList([nn.LSTM(input_size=100, hidden_size=self.lstm_units, batch_first=True, bidirectional=False) for _ in range(output_dim)])
        self.pos_avg_hz_MA_lstm_list = nn.ModuleList([Attention() for _ in range(output_dim)])
        self.output_dim = output_dim
        self.final_pred_layers = nn.ModuleList([nn.Sequential(
                nn.Linear(2*self.lstm_units, 1),
                nn.Sigmoid()
            ) for _ in range(output_dim)])
        self.multihead_attn = nn.MultiheadAttention(100, 1)
        self.layer_norm = nn.LayerNorm(output_dim)   

    def forward(self, dialog, dial_mask):
        self.dialog_model.eval()
        dial_tensors = torch.zeros(dialog.shape[0], self.max_dialog_turns, self.max_len, self.embed_dim).to(self.device)
        for i in range(dialog.shape[0]):
            with torch.no_grad():
                embed = self.dialog_model.encoder(dialog[i], attention_mask=dial_mask[i]).last_hidden_state
            padded = utils.pad_hierarchical_text_sequences(embed, self.embed_dim, self.max_len, self.max_dialog_turns).to(self.device)
            dial_tensors[i] = padded
            del padded, concated, encoder_hidden, decoder_hidden, outputs, dial, mask, label
        padded = dial_tensors.reshape(-1, self.embed_dim, self.max_len, self.max_dialog_turns)
        zcnn = self.zcnn(padded)
        avg_zcnn = self.avg_zcnn(zcnn)
        avg_zcnn = avg_zcnn.permute(0,2,1)
        
        pos_hz_MA_list = [self.pos_hz_MA_list[i](avg_zcnn) for i in range(self.output_dim)]
        pos_hz_MA_lstm_list = [self.pos_hz_MA_lstm_list[i](pos_hz_MA_list[i]) for i in range(self.output_dim)]
        # pdb.set_trace()
        pos_avg_hz_MA_lstm_list = [self.pos_avg_hz_MA_lstm_list[i](pos_hz_MA_lstm_list[i][0]) for i in range(self.output_dim)]

        pos_avg_hz_lstm = torch.cat([pos_avg_hz_MA_lstm_list[i].reshape(-1, 1, self.lstm_units) for i in range(self.output_dim)], dim=1)

        final_preds = []
        for i in range(self.output_dim):
            mask = torch.tensor([True for _ in range(self.output_dim)], dtype=torch.bool).to(self.device)
            mask[i] = False
            target_rep = pos_avg_hz_lstm[:,i:i+1,:]
            source_rep = pos_avg_hz_lstm[:,mask,:]
            attn_w = self.multihead_attn(target_rep.permute(1,0,2), source_rep.permute(1,0,2), source_rep.permute(1,0,2))[0]
            attention_conat = torch.cat((target_rep, attn_w.permute(1,0,2)), dim=-1)
            attention_concat = attention_conat.view(-1, 2*self.lstm_units)
            final_pred = self.final_pred_layers[i](attention_concat)
            final_preds.append(final_pred)
        final_preds.append(sum(final_preds))
        y = torch.cat([final_preds[i] for i in range(self.output_dim)], dim=1).to(self.device)
        
        return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset/')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--ads_train_batch_size', type=int, default=8)
    parser.add_argument('--ads_eval_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--ads_train_epochs', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_name', type=str, default='scoringwoIwC')
    parser.add_argument('--model_name_or_path', type=str, default='t5-small')
    parser.add_argument('--max_turns', type=int, default=7)
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--usr_token', type=str, default='<usr>')
    parser.add_argument('--sys_token', type=str, default='<sys>')
    parser.add_argument('--bos_token', type=str, default='<s>')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--ckpt_name', type=str, default='best_ckpt_epoch=4_valid_loss=0.0147', help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default = "quit!")
    parser.add_argument('--special_eos', type=str, default = "</s>")
    parser.add_argument('--pad_token', type=str, default = '<pad>')

    args = parser.parse_args()
    date = datetime.datetime.now()
    date_txt = date.strftime("%Y%m%d%A%H:%M:%S")
    logger = CreateLogger("scoringwoIwC", f"{args.output_dir}/logs/{args.log_name}{date_txt}.log")
    NGPU = torch.cuda.device_count()
    args.n_gpus = NGPU

    args.local_rank = [i for i in range(args.n_gpus)]
    args.num_workers = NGPU


    trADS = trainADS(args, logger)
    trADS.running_train()
