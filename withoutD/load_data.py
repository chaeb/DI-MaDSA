import os
import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from constants import SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, PHQ_TOKENS

class DialData(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, args, max_len):
        self.tokenizer = tokenizer
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.max_len = max_len
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_turns = args.max_turns
        self.inputs = []
        self.targets = []
        print("Loading PData: {}".format(type_path))

        with open(os.path.join(data_dir, type_path + ".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for dial in data:
            turns = dial['logs']
            hists = []
            emo = dial['emotion']
            for i, turn in enumerate(turns):
                hists.append(self.usr_token + ' ' +  turn['user'].replace('/n', ''))
                target_text = self.emo_token + ' ' + emo + ' ' + self.sys_token + ' ' + turn['system'].replace('/n', '')
                if len(hists) > self.max_turns:
                    hists = hists[-self.max_turns:]
                input_text = self.bos_token + ' ' + ' '.join(hists) 
                hists.append(target_text)
                input_text = self.tokenizer.tokenize(input_text)
                target_text = self.tokenizer.tokenize(target_text)
                if len(input_text) > self.max_len:
                    input_text = input_text[-(self.max_len-2):]
                input_text = [self.bos_token] + input_text
                input_ids = self.tokenizer.encode_plus(input_text, return_tensors = 'pt', padding = 'max_length')
                target_ids = self.tokenizer.encode_plus(target_text, return_tensors = 'pt', padding = 'max_length')
                self.inputs.append(input_ids)
                self.targets.append(target_ids)



    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) :
        source_ids = self.inputs[index]['input_ids'].squeeze()
        src_mask = self.inputs[index]['attention_mask'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()

        return source_ids, src_mask, target_ids, target_mask
        



def get_dataset(tokenizer, type_path, args):
    return DialData(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  args = args, max_len=args.max_len)

