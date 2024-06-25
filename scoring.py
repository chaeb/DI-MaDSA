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
from load_data import get_dataset, get_scoring_dataset
from utils import call_questions, calculate_similarity, Attention, TimeDistributed, MultiHeadAttention
import utils
import pickle
import gzip
from make_logger import CreateLogger

import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import DistributedDataParallelKwargs

import warnings
warnings.filterwarnings("ignore")

class trainADS(nn.Module):
    def __init__(self, args, logger):
        super(trainADS, self).__init__()
        self.args = args
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        self.device =  self.accelerator.device
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        num_new_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.is_main = self.accelerator.is_local_main_process
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token, self.pos_token, self.neg_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id, self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)       
        self.pred = {key: {'0-0':0, '0-1': 0, '0-2':0, '0-3': 0, '1-0':0, '1-1': 0 , '1-2': 0,'1-3': 0, '2-0': 0,'2-1': 0, '2-2': 0,'2-3': 0, '3-0': 0,'3-1': 0,'3-2': 0,'3-3': 0 } for key in ['interest', 'depressed', 'sleep', 'tired', 'appetite', 'failure', 'concentrating', 'moving']}
        if self.args.ckpt_name is None:
            if self.is_main:
                logger.info("****The checkpoint name is not specified****")
                exit()
        else:
            dial_ckpt = f"{self.args.output_dir}/T5models/{self.args.ckpt_name}"
            self.dialog_model = T5ForConditionalGeneration.from_pretrained(dial_ckpt)
            self.dialog_model.resize_token_embeddings(len(self.tokenizer))   
        
            if os.path.isdir(dial_ckpt):
                if self.is_main:
                    if self.args.mode == 'train':
                        logger.info("The training restarts with the specified checkpoint")
                    else:
                        logger.info("The inference will start withthe specified checkpoint")
            else:
                if self.is_main:
                    logger.info("The specified checkpoint does not exist")
                    exit()

        if os.path.isfile(f"{self.args.data_dir}/adswIwCtrain.pickle"):
            if self.is_main:
                logger.info('***** Loading Train Data from pickle *****')
            with gzip.open(f"{self.args.data_dir}/adswIwCtrain.pickle", 'rb') as f:
                train_data = pickle.load(f)
        else:
            train_data = get_scoring_dataset(self.tokenizer, self.dialog_model, "train", self.args)
            if self.is_main:
                with gzip.open(f"{self.args.data_dir}/adswIwCtrain.pickle", 'wb') as f:
                    pickle.dump(train_data, f)
        if os.path.isfile(f"{self.args.data_dir}/adswIwCvalid.pickle"):
            if self.is_main:
                logger.info('***** Loading Valid Data from pickle *****')
            with gzip.open(f"{self.args.data_dir}/adswIwCvalid.pickle", 'rb') as f:
                valid_data = pickle.load(f)
        else:
            valid_data = get_scoring_dataset(self.tokenizer, self.dialog_model, "valid", self.args)
            if self.is_main:
                with gzip.open(f"{self.args.data_dir}/adswIwCvalid.pickle", 'wb') as f:
                    pickle.dump(valid_data, f)
        collate = utils.PaddedDataset(self.pad_id)
        self.train_loader = DataLoader(train_data, batch_size=self.args.ads_train_batch_size, shuffle=True, collate_fn=collate.pad_collate, num_workers=4)
        self.valid_loader = DataLoader(valid_data, batch_size=self.args.ads_eval_batch_size,  shuffle=False, collate_fn=collate.pad_collate,)
        
        self.best_loss = sys.float_info.max
        self.best_qwk = 0
        self.last_epoch = 0
        num_batches = len(self.train_loader)
        args.ads_total_stpes = num_batches * self.args.ads_train_epochs
        args.ads_warmup_steps = int(args.ads_total_stpes * args.warmup_ratio)
        self.dialog_model.eval()
        self.model = ADS(self.dialog_model, self.device)
        ads_ckpt_path = f"{self.args.output_dir}/ADS/{self.args.ads_ckpt_name}.ckpt"
        if os.path.exists(ads_ckpt_path):
            if self.is_main:
                logger.info('***** Loading ADS checkpoint *****')
            ckpt = torch.load(ads_ckpt_path, map_location=self.device)
            pre_trained = ckpt['model_state_dict']
            self.best_loss = ckpt['best_loss']
            self.last_epoch = ckpt['epoch']
            new_model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pre_trained.items() if k in new_model_dict.keys()}
            if len(pretrained_dict) == 0:
                pretrained_dict = {k.replace('module.',''): v for k, v in pre_trained.items() if k.replace('module.','') in new_model_dict.keys()}
            if len(pretrained_dict) == 0:
                logger.info("Cannot load the checkpoint")
                exit()
            new_model_dict.update(pretrained_dict)
            self.model.load_state_dict(new_model_dict)
            del pre_trained, new_model_dict
            torch.cuda.empty_cache()
            if self.args.mode == 'train':
                if self.is_main:
                    logger.info("The training ADS restarts with the specified checkpoint")
            else:
                if self.is_main:
                    logger.info("The inference ADS will start withthe specified checkpoint")
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optim, num_warmup_steps=args.ads_warmup_steps, num_training_steps=args.ads_total_stpes, power=2)
        self.model, self.optim, self.scheduler, self.train_loader, self.valid_loader = self.accelerator.prepare(self.model, self.optim, self.scheduler, self.train_loader, self.valid_loader)

    def running_train(self):
        self.set_seed(self.args.seed)
        if self.is_main:
            logger.info("***** Running ADS training *****")
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.args.ads_train_epochs + start_epoch):
            if self.is_main:
                txt = f"#"*50 + f"Epoch: {epoch}" + "#"*50
                logger.info(txt)
            train_losses = []
            qwks_dict = {}
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_loader, desc = 'Training..', disable = not self.is_main)):
                with self.accelerator.accumulate(self.model):
                    dialog, dial_mask, phq = batch
                    phq = phq[:,:-1]
                    dialog, dial_mask, phq = dialog.to(self.device), dial_mask.to(self.device), phq.to(self.device)
                    y_pred = self.model(dialog, dial_mask)
                    loss = utils.masked_loss_function(y_pred.squeeze(), phq.squeeze())
                    pred_dict = utils.seperate_and_rescale_for_scoring(y_pred)
                    true_dict = utils.seperate_and_rescale_for_scoring(phq)
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
            if self.is_main:
                logger.info(f"Train loss: {train_loss} || Train QWK: {mean_qwk}")

                for key in train_qwk.keys():
                    logger.info(f"Train QWK {key}: {train_qwk[key]}")

            valid_loss, valid_qwk, valid_qwk_dict = self.valid_ads()
            
            if valid_qwk > self.best_qwk:
                self.best_qwk = valid_qwk
            self.accelerator.wait_for_everyone()
            if self.is_main:
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
        if self.is_main:
            logger.info("ADS training finished!")

    def valid_ads(self):
        if self.is_main:
            logger.info("***** Running ADS validation *****")
        self.model.eval()
        valid_losses = []
        qwks_dict = {}
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader, desc = 'Validating..', disable = not self.is_main)):
                dialog, dial_mask, phq = batch
                # for i, p in enumerate(phq):
                #     phq[i] = utils.rescale_score(p)
                phq = phq[:,:-1]
                phq_c = utils.for_classification(phq).to(self.device)
                dialog, dial_mask, phq = dialog.to(self.device), dial_mask.to(self.device), phq.to(self.device)
                y_pred = self.model(dialog, dial_mask)
                # loss = self.criterion(y_pred.squeeze(), phq_c.squeeze().long())
                loss = utils.masked_loss_function(y_pred.squeeze(), phq.squeeze())
                pred_dict = utils.seperate_and_rescale_for_scoring(y_pred, is_phq=True)
                true_dict = utils.seperate_and_rescale_for_scoring(phq, is_phq=True)
                qwk_dict = {key: utils.kappa(true_dict[key], pred_dict[key]) for key in pred_dict.keys()}
                for key in true_dict.keys():
                    if key != 'total':
                        for t, p in zip(true_dict[key], pred_dict[key]):
                            self.pred[key][f'{int(t)}-{int(p)}'] += 1
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
        self.max_dialog_turns = 40
        self.dropout_prob = 0.3
        self.cnn_filters = 100
        self.cnn_kernerl_size = 5
        self.max_len = 512
        self.embed_dim = 512
        self.lstm_units = 100
        self.dialog_model = dialog_model
        self.device = device
        self.zcnn = TimeDistributed(nn.Conv1d(in_channels = 512, out_channels = self.cnn_filters, kernel_size=self.cnn_kernerl_size, padding=0))
        self.avg_zcnn = TimeDistributed(Attention(self.device))
        self.pos_hz_MA_list = nn.ModuleList([MultiHeadAttention(self.device, self.embed_dim, 100, 2) for _ in range(output_dim)])
        self.pos_hz_MA_lstm_list = nn.ModuleList([nn.LSTM(input_size=100, hidden_size=self.lstm_units, batch_first=True, bidirectional=False) for _ in range(output_dim)])
        self.pos_avg_hz_MA_lstm_list = nn.ModuleList([Attention(self.device) for _ in range(output_dim)])
        self.output_dim = output_dim
        self.final_pred_layers = nn.ModuleList([nn.Sequential(
                nn.Linear(2*self.lstm_units, 1),
                nn.Sigmoid()
            ) for _ in range(output_dim)])
        self.multihead_attn = nn.MultiheadAttention(100, 1)
        self.layer_norm = nn.LayerNorm(output_dim)   
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'dialog_model' in name:
                pass
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

                
    def forward(self, dialog, dial_mask):
        self.dialog_model.eval()
        dial_tensors = torch.zeros(dialog.shape[0], self.max_dialog_turns, self.max_len, self.embed_dim).to(self.device)
        for i in range(dialog.shape[0]):
            with torch.no_grad():
                embed = self.dialog_model.encoder(dialog[i], attention_mask=dial_mask[i]).last_hidden_state.detach()
            padded = utils.pad_hierarchical_text_sequences(embed, self.embed_dim, self.max_len, self.max_dialog_turns).to(self.device)
            dial_tensors[i] = padded
            del padded
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
    parser.add_argument('--ads_train_batch_size', type=int, default=128)
    parser.add_argument('--ads_eval_batch_size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--ads_train_epochs', type=int, default=20)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--log_name', type=str, default='RegwIwCLR')
    parser.add_argument('--model_name_or_path', type=str, default='t5-small')
    parser.add_argument('--max_turns', type=int, default=7)
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--mode', type=str, default='valid')
    parser.add_argument('--usr_token', type=str, default='<usr>')   
    parser.add_argument('--sys_token', type=str, default='<sys>')
    parser.add_argument('--bos_token', type=str, default='<s>')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--ckpt_name', type=str, default='best_ckpt_epoch=10_valid_loss=0.022', help="The name of the trained checkpoint. (without extension)")
    parser.add_argument('--end_command', type=str, default = "quit!")
    parser.add_argument('--special_eos', type=str, default = "</s>")
    parser.add_argument('--pad_token', type=str, default = '<pad>')
    # parser.add_argument('--ads_ckpt_name', type=str, required=False)
    parser.add_argument('--ads_ckpt_name', type=str, default='best_ckpt_epoch=20_valid_loss=0.0205', help="The name of the trained checkpoint. (without extension)")
    args = parser.parse_args()
    date = datetime.datetime.now()
    date_txt = date.strftime("%Y%m%d%A%H:%M:%S")
    logger = CreateLogger("RegwIwC", f"{args.output_dir}/logs/{args.log_name}{date_txt}.log")
    NGPU = torch.cuda.device_count()
    args.n_gpus = NGPU

    args.local_rank = [i for i in range(args.n_gpus)]
    args.num_workers = NGPU
    args.ads_train_batch_size = int(args.ads_train_batch_size / args.n_gpus)
    args.ads_eval_batch_size = int(args.ads_eval_batch_size / args.n_gpus)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps / args.n_gpus)
    args.lr = args.lr * args.n_gpus

    trADS = trainADS(args, logger)
    if args.mode == 'train':
        trADS.running_train()
    else:
        trADS.valid_ads()
        with open('./pred.json', 'w') as f:
            json.dump(trADS.pred, f)
