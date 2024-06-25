from sklearn.metrics.pairwise import cosine_similarity

def call_questions():
    return {'depressed':["How often feeling down, depressed, irritable, or hopeless in the past two weeks?"],
            'interest':["How often little interest or pleasure in doing things in the past two weeks?"],
            'sleep':["How often trouble falling or staying asleep, or sleeping too much in the past two weeks?"],
            'tired':["How often feeling tired or having little energy in the past two weeks?"],
            'appetite':["How often poor appetite or overeating in the past two weeks?"],
            'failure': ["How often feeling bad about yourself or that you are a failure in the past two weeks?", "How often have you let yourself or your family down in the past two weeks?"],
            'concentrating': ["How often have you had trouble concentrating on things in the past two weeks?"],
            'moving': ["How often moving or speaking so slowly that other people could have noticed in the past two weeks?"]}

def calculate_similarity(text1,text2):
    
    return cosine_similarity(text1, text2)[0][0]



import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn

score_mapping = {0: 'interest', 1: 'depressed', 2: 'sleep', 3: 'tired', 4:'appetite', 5: 'failure', 6: 'concentrating', 7: 'moving', 8: 'total'}
reverse_mapping = {'interest': 0, 'depressed': 1, 'sleep': 2, 'tired': 3, 'appetite': 4, 'failure': 5, 'concentrating': 6, 'moving': 7, 'total': 8}

def get_score_vector_positions():
    return deepcopy(reverse_mapping)

def get_min_max_score():
    return {
            'phq': {0: (0, 3), 1: (0, 3), 2: (0, 3), 3: (0, 3), 4: (0, 3), 5: (0, 3), 6: (0, 3), 7: (0, 3), 8: (0, 24)},
            'ptsd': {0: (17,85)}
            }


def get_scaled_down_scores(scores, is_phq = True):
    if is_phq:
        score_positions = deepcopy(score_mapping)
        min_max_scores = get_min_max_score()['phq']
        rescaled_score_vector = torch.zeros(len(score_positions))
        for k, score in scores.items():
            min_score, max_score = min_max_scores[reverse_mapping[k]]
            rescaled_score = (score - min_score) / (max_score - min_score)
            rescaled_score_vector[reverse_mapping[k]] = rescaled_score
        return rescaled_score_vector
    else:
        min_score, max_score = get_min_max_score()['ptsd'][0]
        rescaled_score = (scores - min_score) / (max_score - min_score)
        return rescaled_score
    
def get_scaled_up_scores(scores, is_phq = True):
    if is_phq:
        score_positions = deepcopy(score_mapping)
        min_max_scores = get_min_max_score()['phq']
        rescaled_score_vector = [-1]*len(score_positions)
        for k, score in enumerate(scores):
            min_score, max_score = min_max_scores[k]
            rescaled_score = score * (max_score - min_score) + min_score
            rescaled_score_vector[k] = int(rescaled_score)
        return rescaled_score_vector
    else:
        min_score, max_score = get_min_max_score()['ptsd'][0]
        rescaled_score = scores * (max_score - min_score) + min_score
        return rescaled_score
    

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        input_shape = input_seq.shape

        X = input_seq.reshape((-1, ) + input_shape[2:])  # (nb_samples * timesteps, ...)
        y = self.module(X)  # (nb_samples * timesteps, ...)
        # (nb_samples, timesteps, ...)
        if type(y) == tuple: y = y[0]   # lstm
        y = y.reshape((-1, input_shape[1]) + y.shape[1:])
        return y

class Attention(nn.Module):
    def __init__(self, device, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self.built = False
        self.device = device

    def build(self, input_shape):
        init_val_v = (torch.randn(input_shape[2]) * self.init_stdev).float().to(self.device)
        self.att_v = nn.Parameter(init_val_v).to(self.device)
        init_val_W = (torch.randn(input_shape[2], input_shape[2]) * self.init_stdev).float().to(self.device)
        self.att_W = nn.Parameter(init_val_W).to(self.device)
        self.built = True

    def forward(self, x, mask=None):
        if not self.built:
            self.build(x.shape)
        y = torch.matmul(x, self.att_W)
        if not self.activation:
            weights = torch.tensordot(self.att_v, y, dims=([0], [2]))
        elif self.activation == 'tanh':
            weights = torch.tensordot(self.att_v, torch.tanh(y), dims=([0], [2]))
        weights = torch.softmax(weights, dim=1)
        out = x * weights.unsqueeze(2).expand_as(x)
        if self.op == 'attsum':
            out = torch.sum(out, dim=1)
        elif self.op == 'attmean':
            out = torch.sum(out, dim=1) / mask.sum(dim=1, keepdim=True)
        return out.float().to(self.device)  
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, inputs, mask=None):
        return None
    

class MultiHeadAttention(nn.Module):
    def __init__(self, device, input_dim,  embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim  # d_model
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0
        self.device = device
        self.projection_dim = embedding_dim // num_heads
        self.query_dense = nn.Linear(input_dim, embedding_dim).to(self.device)
        self.key_dense = nn.Linear(input_dim, embedding_dim).to(self.device)
        self.value_dense = nn.Linear(input_dim, embedding_dim).to(self.device)
        self.dense = nn.Linear( self.embedding_dim, embedding_dim).to(self.device)
        

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        depth = torch.tensor(self.embedding_dim // self.num_heads, dtype=torch.float32)
        logits = matmul_qk / torch.sqrt(depth)
        attention_weights = nn.functional.softmax(logits, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.transpose(1, 2)

    def forward(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = inputs.size(0)

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = scaled_attention.transpose(1, 2)

        # (batch_size, seq_len, embedding_dim)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.embedding_dim)
        outputs = self.dense(concat_attention)
        #outputs = torch.sum(outputs, dim=1) # 1011 추가
        return outputs


def pad_hierarchical_text_sequences(sequences, embed_dim, max_len, max_sentnum):
    X = torch.zeros((max_sentnum, max_len, embed_dim))
    if len(sequences) > max_sentnum:
        sequences = sequences[-max_sentnum:]
    for i, sentence in enumerate(sequences):
        X[i] = sentence
    return X


def masked_loss_function(y_true, y_pred):
    mask_value = -1
    mask = torch.tensor(y_true != mask_value, dtype=torch.float32)
    mse = nn.MSELoss()
    return mse(y_true * mask, y_pred * mask)

from sklearn.metrics import confusion_matrix
from six import string_types

def rescale_score(scores):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_score()['phq']
    rescaled_score_vector = torch.zeros(len(score_vector_positions))
    for k, score in enumerate(scores):
        min_score, max_score = min_max_scores[k]
        rescaled_score = score * (max_score - min_score) + min_score
        rescaled_score_vector[k] = int(rescaled_score)
    return rescaled_score_vector

def get_dict_score(scores):
    min_max_scores = get_min_max_score()['phq']
    score_dict = {}
    for k, score in enumerate(scores):
        keys = list(min_max_scores.keys())
        total = 0
        for att in keys[:-1]:
            try:
                score_dict[score_mapping[att]].append(int(score[att].item()))
            except:
                score_dict[score_mapping[att]] = [int(score[att].item())]
            total += int(score[att].item())
        try:
            score_dict['total'].append(total)
        except:
            score_dict['total'] = [total]
    return score_dict

def for_classification(scores):
    temp = torch.zeros((len(scores), 8, 4), dtype=torch.long)
    for i, score in enumerate(scores):
        for j, idx in enumerate(score):
            idx = int(idx.item())
            temp[i, j, idx] = 1
    return temp
        



def seperate_and_rescale_for_scoring(scores, is_phq = True):
    score_vector_positions = get_score_vector_positions()
    individual_scores = {}
    if is_phq:
        min_max_scores = get_min_max_score()['phq']
    else:
        min_max_scores = get_min_max_score()['ptsd']
    for i, score in enumerate(scores):
        keys = list(min_max_scores.keys())
        total = 0
        for att in keys[:-1]:
            min_score = min_max_scores[att][0]
            max_score = min_max_scores[att][1]
            rescaled_score = score[att] * (max_score - min_score) + min_score
            total += np.round(rescaled_score.item())
            try:
                individual_scores[score_mapping[att]].append(np.round(rescaled_score.item()))
            except:
                individual_scores[score_mapping[att]] = [np.round(rescaled_score.item())]
        try:
            individual_scores['total'].append(total)
        except:
            individual_scores['total'] = [total]
    return individual_scores

def kappa(y_true, y_pred, weights = "quadratic", allow_off_by_one=False):
    qwks = deepcopy(reverse_mapping)
    
    y_true = [int(np.round(float(y))) for y in y_true]
    y_pred = [int(np.round(float(y))) for y in y_pred]

    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))

    num_scored_items = float(len(y_true))

    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''

    if weights is None: 
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))
    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    observed = observed / num_scored_items

    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))
    return k

from torch.utils.data import Dataset

class PaddedDataset(Dataset):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
            dials, dial_masks,  phqs = [], [], []
            for idx, seqs in enumerate(batch):
                dials_tensor = torch.randint(0,10,(len(seqs[0]), 512))
                dial_masks_tensor = torch.randint(0,10,(len(seqs[1]), 512))
                for i in range(len(seqs[0])):
                    if len(seqs[0][i]) > 512:
                        seqs[0][i] = seqs[0][i][:512]
                    if len(seqs[1][i]) > 512:
                        seqs[1][i] = seqs[1][i][:512]
                    dials_tensor[i, :len(seqs[0][i])] = torch.LongTensor(seqs[0][i])
                    dial_masks_tensor[i, :len(seqs[1][i])] = torch.LongTensor(seqs[1][i])
                    dials_tensor[i, len(seqs[0][i]):] = self.pad_id
                    dial_masks_tensor[i, len(seqs[1][i]):] = self.pad_id
                dials.append(dials_tensor)
                dial_masks.append(dial_masks_tensor)
                phqs.append(torch.FloatTensor(seqs[2]))  

            dials = torch.nn.utils.rnn.pad_sequence(dials, batch_first=True, padding_value=self.pad_id)
            dial_masks = torch.nn.utils.rnn.pad_sequence(dial_masks, batch_first=True, padding_value=self.pad_id)
            phqs = torch.stack(phqs)

            
            return dials, dial_masks, phqs
    
    def pad_collate_test(self, batch):
            dials, dial_masks, phqs = [], [], []
            for idx, seqs in enumerate(batch):
                dials_tensor = torch.randint(0,10,(len(seqs[0]), 512))
                dial_masks_tensor = torch.randint(0,10,(len(seqs[1]), 512))
                for i in range(len(seqs[0])):
                    if len(seqs[0][i]) > 512:
                        seqs[0][i] = seqs[0][i][:512]
                    if len(seqs[1][i]) > 512:
                        seqs[1][i] = seqs[1][i][:512]
                    dials_tensor[i, :len(seqs[0][i])] = torch.LongTensor(seqs[0][i])
                    dial_masks_tensor[i, :len(seqs[1][i])] = torch.LongTensor(seqs[1][i])
                    dials_tensor[i, len(seqs[0][i]):] = self.pad_id
                    dial_masks_tensor[i, len(seqs[1][i]):] = self.pad_id
                dials.append(dials_tensor)
                dial_masks.append(dial_masks_tensor)
                phqs.append(torch.LongTensor([seqs[2]])) 

            dials = torch.nn.utils.rnn.pad_sequence(dials, batch_first=True, padding_value=self.pad_id)
            dial_masks = torch.nn.utils.rnn.pad_sequence(dial_masks, batch_first=True, padding_value=self.pad_id)
            
            phqs = torch.stack(phqs)
            
            return dials, dial_masks, phqs
    


class PaddedDAICDataset(Dataset):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
            dials, dial_masks, phqs, deps = [], [], [], []
            for idx, seqs in enumerate(batch):
                dials_tensor = torch.randint(0,10,(len(seqs[0]), 512))
                dial_masks_tensor = torch.randint(0,10,(len(seqs[1]), 512))
                for i in range(len(seqs[0])):
                    if len(seqs[0][i]) > 512:
                        seqs[0][i] = seqs[0][i][:512]
                    if len(seqs[1][i]) > 512:
                        seqs[1][i] = seqs[1][i][:512]
                    dials_tensor[i, :len(seqs[0][i])] = torch.LongTensor(seqs[0][i])
                    dial_masks_tensor[i, :len(seqs[1][i])] = torch.LongTensor(seqs[1][i])
                    dials_tensor[i, len(seqs[0][i]):] = self.pad_id
                    dial_masks_tensor[i, len(seqs[1][i]):] = self.pad_id
                dials.append(dials_tensor)
                dial_masks.append(dial_masks_tensor)
                phqs.append(torch.FloatTensor(seqs[2]))
                deps.append(torch.LongTensor([seqs[3]]))

            dials = torch.nn.utils.rnn.pad_sequence(dials, batch_first=True, padding_value=self.pad_id)
            dial_masks = torch.nn.utils.rnn.pad_sequence(dial_masks, batch_first=True, padding_value=self.pad_id)
            phqs = torch.stack(phqs)
            deps = torch.stack(deps)

            return dials, dial_masks, phqs, deps
    
    def pad_collate_test(self, batch):
            dials, dial_masks, phqs, deps = [], [], [], []
            for idx, seqs in enumerate(batch):
                dials_tensor = torch.randint(0,10,(len(seqs[0]), 512))
                dial_masks_tensor = torch.randint(0,10,(len(seqs[1]), 512))
                for i in range(len(seqs[0])):
                    if len(seqs[0][i]) > 512:
                        seqs[0][i] = seqs[0][i][:512]
                    if len(seqs[1][i]) > 512:
                        seqs[1][i] = seqs[1][i][:512]
                    dials_tensor[i, :len(seqs[0][i])] = torch.LongTensor(seqs[0][i])
                    dial_masks_tensor[i, :len(seqs[1][i])] = torch.LongTensor(seqs[1][i])
                    dials_tensor[i, len(seqs[0][i]):] = self.pad_id
                    dial_masks_tensor[i, len(seqs[1][i]):] = self.pad_id
                dials.append(dials_tensor)
                dial_masks.append(dial_masks_tensor)
                phqs.append(torch.LongTensor([seqs[2]]))
                deps.append(torch.LongTensor([seqs[3]]))

            dials = torch.nn.utils.rnn.pad_sequence(dials, batch_first=True, padding_value=self.pad_id)
            dial_masks = torch.nn.utils.rnn.pad_sequence(dial_masks, batch_first=True, padding_value=self.pad_id)
            phqs = torch.stack(phqs)
            deps = torch.stack(deps)
            
            return dials, dial_masks, phqs, deps