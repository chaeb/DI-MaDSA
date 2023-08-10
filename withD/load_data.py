import os
import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from constants import SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, PHQ_TOKENS
import utils
from tqdm import tqdm

class DialData(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, args, max_len):
        self.tokenizer = tokenizer
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token, self.pos_token, self.neg_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id, self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.max_len = max_len
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_turns = args.max_turns
        self.inputs = []
        self.targets = []
        print("Loading PData: {}".format(type_path))

        with open(os.path.join(data_dir, type_path + ".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for dial in tqdm(data):
            turns = dial['logs']
            hists = []
            emo = dial['emotion']
            for i, turn in enumerate(turns):
                hists.append(self.usr_token + ' ' +  turn['user'].replace('/n', ''))
                target_text = self.emo_token + ' ' + '<'+emo+'>' + ' ' + self.sys_token + ' ' + turn['system'].replace('/n', '')
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


class ScoreDataset(Dataset):
    def __init__(self, tokenizer, dialmodel, data_dir, type_path, args, max_len=512):
        self.tokenizer = tokenizer
        self.bos_token, self.eos_token, self.usr_token, self.sys_token, self.pad_token, self.emo_token, self.pos_token, self.neg_token = SPECIAL_TOKENS
        self.bos_id, self.eos_id, self.usr_id, self.sys_id, self.pad_id, self.emo_id, self.pos_id, self.neg_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        self.max_len = max_len
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_turns = args.max_turns
        self.dials = []
        self.attention_masks = []
        self.dials_label = []
        self.phqs = []
        self.max_dialog_num = 0
        self.dialmodel = dialmodel
        self.dialmodel.eval()

        print("Loading Scoring Dataset: {}".format(type_path))

        with open(os.path.join(data_dir, type_path + ".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        if type_path == 'train':
            for dial in tqdm(data):
                turns = dial['logs']
                hists = []
                attention_masks = []
                phq_scores = {}
                emo = dial['emotion']
                for att in dial['score'].keys():
                    phq_scores[att] = int(dial['score'][att])
                phq_scores['total'] = sum(phq_scores.values())
                rescaled_phq  = utils.get_scaled_down_scores(phq_scores, is_phq=True)
                for i, turn in enumerate(turns):
                    input_text = self.usr_token + ' ' +  turn['user'].replace('/n', '')
                    target_text = self.emo_token + ' ' + '<'+emo+'>' + ' ' + self.sys_token + ' ' + turn['system'].replace('/n', '')
                    input_text = self.bos_token + ' ' + input_text
                    input_text = self.tokenizer.tokenize(input_text)
                    target_text = self.tokenizer.tokenize(target_text)
                    input_ids = self.tokenizer.encode_plus(input_text, return_tensors = 'pt', padding = 'max_length')
                    target_ids = self.tokenizer.encode_plus(target_text, return_tensors = 'pt', padding = 'max_length')
                    hists.append(input_ids['input_ids'].squeeze())
                    hists.append(target_ids['input_ids'].squeeze())
                    attention_masks.append(input_ids['attention_mask'].squeeze())
                    attention_masks.append(target_ids['attention_mask'].squeeze())
                self.dials.append(hists)
                self.attention_masks.append(attention_masks)
                self.phqs.append(rescaled_phq)
        
        else:
            for dial in tqdm(data):
                turns = dial['logs']
                hists = []
                dial_ids = []
                attention_masks = []
                phq_scores = {}
                for att in dial['score'].keys():
                    phq_scores[att] = int(dial['score'][att])
                phq_scores['total'] = sum(phq_scores.values())
                rescaled_phq  = utils.get_scaled_down_scores(phq_scores, is_phq=True)
                batch_input = []
                batch_mask = []
                curr_users = []
                for i, turn in enumerate(turns):
                    hists.append(self.usr_token + ' ' +  turn['user'].replace('/n', ''))
                    if len(hists) > self.max_turns:
                        hists = hists[-self.max_turns:]
                    input_text = self.bos_token + ' ' + ' '.join(hists)
                    input_text = self.tokenizer.tokenize(input_text)
                    input_ids = self.tokenizer.encode_plus(input_text, return_tensors = 'pt', padding = 'max_length')
                    batch_input.append(input_ids['input_ids'].squeeze().tolist())
                    batch_mask.append(input_ids['attention_mask'].squeeze().tolist())   
                    current_user = self.usr_token + ' ' + turn['user'].replace('/n', '')
                    current_user_ids = self.tokenizer.encode_plus(current_user, return_tensors = 'pt', padding = 'max_length')
                    curr_users.append(current_user_ids)
                    if len(batch_input) == 5:
                        with torch.no_grad():
                            gens = self.dialmodel.generate(torch.tensor(batch_input), attention_mask=torch.tensor(batch_mask), max_length=self.max_len, num_beams=5, early_stopping=True)
                            for j in range(len(gens)):
                                dial_ids.append(curr_users[j]['input_ids'].squeeze())
                                attention_masks.append(curr_users[j]['attention_mask'].squeeze())
                                gen = self.tokenizer.decode(gens[j], skip_special_tokens=True)
                                gen_ids = self.tokenizer.encode_plus(gen, return_tensors = 'pt', padding = 'max_length')
                                dial_ids.append(gen_ids['input_ids'].squeeze())
                                attention_masks.append(gen_ids['attention_mask'].squeeze())
                        batch_input = []
                        batch_mask = []
                        curr_users = []
                    hists.append(self.sys_token + ' ' + turn['system'].replace('/n', ''))
                if len(batch_input):
                    with torch.no_grad():
                        gens = self.dialmodel.generate(torch.tensor(batch_input), attention_mask=torch.tensor(batch_mask), max_length=self.max_len, num_beams=5, early_stopping=True)
                        for j in range(len(gens)):
                            dial_ids.append(curr_users[j]['input_ids'].squeeze())
                            attention_masks.append(curr_users[j]['attention_mask'].squeeze())
                            gen = self.tokenizer.decode(gens[j], skip_special_tokens=True)
                            gen_ids = self.tokenizer.encode_plus(gen, return_tensors = 'pt', padding = 'max_length')
                            dial_ids.append(gen_ids['input_ids'].squeeze())
                            attention_masks.append(gen_ids['attention_mask'].squeeze())
                self.dials.append(dial_ids)
                self.attention_masks.append(attention_masks)
                self.phqs.append(rescaled_phq)
    
    
    def __len__(self):
        return len(self.dials)

    def __getitem__(self, index) :
        dial = self.dials[index]
        dial_mask = self.attention_masks[index]
        phq = self.phqs[index]
        return dial, dial_mask, phq
    

def get_scoring_dataset(tokenizer, dialmodel, type_path, args):
    return ScoreDataset(tokenizer=tokenizer, dialmodel=dialmodel, data_dir=args.data_dir, type_path=type_path,  args = args, max_len=args.max_len)

