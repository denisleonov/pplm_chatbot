import pandas as pd
import json

import random
import torch
import numpy as np
import re
from unidecode import unidecode
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from torchvision.transforms import Compose
import config as cfg
import yaml
from data import TextTransforms
from pathlib import Path
import torch
DATA_PARTS = ['id', 'sentence', 'classes']
SQUEEZE_REPEATED = [r"\'\`\!\?\."]




def parse_toxic_data(path):
    """Function to extract data from files in folder to classification dataset specified in config.py.

    :param: path: path to data csv file
    :return: 
        data: list [dict with keys: "sentences", "classes"]
    """
    def extract_idx(line):
        '''
        Function to extract number of line and split line to (N_line, line)
        for expample: "1 your persona: ..." -> (1, "your persona: ...") 
        '''
        return int((line['id'], line['comment_text']))


    def extract_sentence(line):
        '''
        Function to split text line into two sentences: (self_sentence, other_sentence)
        '''
        return line['comment_text']

    def extract_classes(line):
        return np.array(line[2:])

    lines = pd.DataFrame(columns=DATA_PARTS)# array of dicts with DATA_PARTS keys
    print(path)
    data = pd.read_csv(path)
    lines['sentence'] = data.apply(extract_sentence, axis=1)
    lines['classes'] = data.apply(extract_classes, axis=1)
    print('ok')
    return lines

def parse_col_data(path, text_col='sentence', class_col='classes', config_file=None, sep='\t'):
    """Function to extract data from files in folder to classification dataset specified in config.py.

    :param: path: path to data csv file
    :return:
        data: list [dict with keys: "sentences", "classes"]
    """
    lines = pd.DataFrame(columns=DATA_PARTS)  # array of dicts with DATA_PARTS keys
    data = pd.read_csv(path, sep=sep)
    #print(data.head())
    print(path)
    lines['sentence'] = data[text_col]
    lines['classes'] = data[class_col]
    if config_file is not None:
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
        lines['classes'] = lines['classes'].apply(lambda x: config['classes'][x])
    print('ok')
    return lines

def parse_standart_data(path):
    return parse_col_data(path)

def parse_decade_data(path):
    return parse_col_data(path, text_col='text', class_col='decade', config_file='head_configs/decade.yaml')

def parse_age_data(path):
    return parse_col_data(path, text_col='text', class_col='age', config_file='head_configs/age.yaml')

def parse_gender_data(path):
    return parse_col_data(path, text_col='text', class_col='gender', config_file='head_configs/gender.yaml')

def parse_sign_data(path):
    return parse_col_data(path, text_col='text', class_col='sign', config_file='head_configs/sign.yaml')

def parse_topic_data(path):
    return parse_col_data(path, text_col='text', class_col='topic', config_file='head_configs/topics_extended.yaml')

class AttrDataset(Dataset):
    """Class to work with jigsaw toxic data while training classifier"""

    def __init__(self, path, tokenizer, special_ids,
                 transforms, parse_function, type='multilabel'):
        self.special_ids = special_ids
        self.type = type
        self.tokenizer = tokenizer
        self.data = parse_function(path)
        self.transforms = transforms
    #print(self.transforms)
        
    def __len__(self):
        """Returns num of examples."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """ Function to get single piece of data
        
        :param idx: index of data element
        :return:
            sentence, classes
        """

        sample = self.data.loc[idx,]
        #print(sample)
        ids = []
        tt_ids = []
        piece_spec_id: int
        
        text_piece = sample['sentence'].replace('\n', '')
        piece_spec_id = self.special_ids.self
        piece_ids = [piece_spec_id] + self.tokenizer.encode(text_piece, add_special_tokens=True,
                                                            max_length=(cfg.MAX_LEN-1))
        piece_tt_ids = [piece_spec_id] * min(len(piece_ids), cfg.MAX_LEN)
        #print('in get', len(piece_ids))
        #print(sample['classes'])
        return piece_ids, piece_tt_ids, sample['classes'].astype('long')


def create_loader(dataset, batch_size, collate_fn):
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS
    )


def get_dataloaders(tokenizer, special_ids, batch_size, ds_files, dataset='toxic'):
    print("===> Set up dataloaders")
    def pad(seq, max_len, pad_elem=special_ids.pad):
        """ Func to pad sequence """
        return seq + [pad_elem] * max(0, max_len - len(seq))

    def train_collate_fn(batch_data):
        """ Function to pull train batch
        :param batch_data: list of `PersonaChatTrainDataset` examples
        :return:
            ids, tt_ids, classes, masks, pos_ids | torch.Size([batch_size, max_examples_len])
        """

        batch_ids, batch_tt_ids, batch_classes = list(zip(*batch_data))
        batch_len = max(map(len, batch_ids))
        # pad ids and create masks/pos_ids
        batch_ids = [pad(ids, batch_len) for ids in batch_ids]
        batch_tt_ids = [pad(tt_ids, batch_len) for tt_ids in batch_tt_ids]
        batch_masks = [pad([1] * l, batch_len, 0) for l in map(len, batch_ids)]
        batch_pos_ids = [list(range(batch_len))] * len(batch_data)
        return [
            torch.tensor(data_prt, dtype=torch.long)
            for data_prt in (batch_ids, batch_tt_ids, batch_classes, batch_masks, batch_pos_ids)
        ]

    def test_collate_fn(batch_data):
        batch_ids, batch_tt_ids, batch_classes = list(zip(*batch_data))
        batch_lens = list(map(len, batch_ids))

        # pad ids and create masks/pos_ids
        batch_ids = [torch.tensor(ids, dtype=torch.long) for ids in batch_ids]
        batch_tt_ids = [torch.tensor(tt_ids, dtype=torch.long) for tt_ids in batch_tt_ids]
        batch_masks = [torch.tensor([1] * l, dtype=torch.long) for l in batch_lens]
        batch_pos_ids = [torch.tensor(list(range(l)), dtype=torch.long) for l in batch_lens]
        batch_classes = [torch.tensor(classes, dtype=torch.float).unsqueeze(0) for classes in batch_classes]
        return batch_ids, batch_tt_ids, batch_classes, batch_masks, batch_pos_ids



    train_transforms = Compose([TextTransforms.random_lcut,
                 TextTransforms.standardize])
    if dataset in ['toxic']:
        parse_function = parse_toxic_data
        type = 'multilabel'

    elif dataset in ['sentiment', 'clickbait', 'subject', 'categories']:
        parse_function = parse_standart_data
        type = 'multiclass'

    elif dataset == 'decade':
        parse_function = parse_decade_data
        type = 'multiclass'

    elif dataset == 'age':
        parse_function = parse_age_data
        type = 'multiclass'
    elif dataset == 'gender':
        parse_function = parse_gender_data
        type = 'multiclass'
    elif dataset == 'topics_extended':
        parse_function = parse_topic_data
        type = 'multiclass'
    else:
        raise Exception('This attribute is not supported yet!')
    return [
        create_loader(
            dataset=ds_class(files, tokenizer, special_ids, transforms, parse_function, type),
            batch_size=batch_size, collate_fn=collate_fn,
        )
        for ds_class, files, transforms, collate_fn
        in zip(
            [AttrDataset, AttrDataset, AttrDataset],
            [ds_files['train'], ds_files['valid'], ds_files['test']],
            [train_transforms, train_transforms, train_transforms],
            [train_collate_fn, test_collate_fn, test_collate_fn],
        )
    ]



if __name__ == "__main__":

    from collections import namedtuple
    from transformers import GPT2Tokenizer
    special_dct = {
        'info': '<info>',
        'self': '<self>',
        'other': '<other>',
        'pad': '<pad>',
    }

    SpecialIdsClass = namedtuple('special_ids', list(special_dct.keys()))
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({'additional_special_tokens': list(special_dct.values())})
    spec_ids = SpecialIdsClass(**{k: tok.encode(v)[0] for k, v in special_dct.items()})

    transforms = Compose([TextTransforms.random_lcut, TextTransforms.standardize])
    dataset = 'toxic'
    text_piece = "You might like to consider that I don't give a shit what you do or think.".replace('\n', '').lower()
    piece_spec_id = spec_ids.self
    ids = [piece_spec_id] + tok.encode(text_piece, add_special_tokens=True,
                                                        max_length=(cfg.MAX_LEN - 1))
    tt_ids = [piece_spec_id] * min(len(ids), cfg.MAX_LEN)
    classes = np.array([1,0,1,0,1,0]).astype('long')
    l = len(ids)

    # pad ids and create masks/pos_ids
    ids = torch.tensor(ids, dtype=torch.long)
    tt_ids = torch.tensor(tt_ids, dtype=torch.long)
    mask = torch.tensor([1] * l, dtype=torch.long)
    pos_ids = torch.tensor(list(range(l)), dtype=torch.long)
    classes = torch.tensor(classes, dtype=torch.float).unsqueeze(0)
