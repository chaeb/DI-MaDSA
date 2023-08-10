import pickle
import json
from load_datawoC import DialData, get_dataset
from constants import SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../dataset/')
parser.add_argument('--output_dir', type=str, default='./outputs')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--max_output_length', type=int, default=128)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=16)
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

tokenizer = T5Tokenizer.from_pretrained('t5-small')
num_new_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
model = T5ForConditionalGeneration.from_pretrained('t5-small')
args.max_len = min(512, model.config.n_positions)
vocab = tokenizer.get_vocab()
args.vocab_size = len(vocab)
model.resize_token_embeddings(len(vocab))
bos_token, eos_token, usr_token, sys_token, pad_token, emo_token, pos_token, neg_token = SPECIAL_TOKENS
bos_id, eos_id, usr_id, sys_id, pad_id, emo_id, pos_id, neg_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

train_set = get_dataset(tokenizer, 'train', args)
valid_set = get_dataset(tokenizer, 'valid', args)

import gzip

with gzip.open('../dataset/train_ids_woC.pickle', 'wb') as f:
    pickle.dump(train_set, f)

with gzip.open('../dataset/valid_ids_woC.pickle', 'wb') as f:
    pickle.dump(valid_set, f)