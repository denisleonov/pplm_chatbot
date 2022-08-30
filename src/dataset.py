import random
import math
import torch

from functools import partial
from itertools import zip_longest, product
from operator import add
from typing import List, Tuple

from src.dataset_parsers import Parsers, Dialogue


"""
FROM GPT2 FORWARD
if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
"""
class DialogueDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 tokenizer,
                 model_type,
                 mode='train',
                 train_size=0.99,
                 use_topic=False,
                 use_sequence_bucketing=False,
                 context_max_len=256,
                 response_max_len=100,
                 debug=False,
                 seed=42,
                 use_taskmaster=True,
                 use_dailydialog=True,
                 use_personachat=True,
                 use_dailydialog_emotions=False
                ):
        """
        Add special tokens to tokenizer before usage.
        >> tokenizer.add_special_tokens({'additional_special_tokens': ['[CONTEXT]', '[RESPONSE]', '[ALICE]', '[BOB]']})
        >> model.resize_token_embeddings(len(tokenizer))
        """
        assert model_type in ['gpt2', 'bart']
        assert mode in ['train', 'valid', None]
        assert 'additional_special_tokens' in tokenizer.special_tokens_map, 'Add special tokens before usage.'

        assert use_dailydialog is True or use_personachat is True or use_taskmaster is True

        if use_dailydialog_emotions is True:
            assert use_dailydialog and use_taskmaster == use_personachat == False, 'You cannot use emotions only with dd dataset.'

        self.use_taskmaster = use_taskmaster
        self.use_dailydialog = use_dailydialog
        self.use_personachat = use_personachat
        self.use_dailydialog_emotions = use_dailydialog_emotions

        self.context_max_len = context_max_len
        self.response_max_len = response_max_len
        self.model_type = model_type
        self.use_topic = use_topic
        self.use_sequence_bucketing = use_sequence_bucketing
        self.mode = mode
        self.train_size = train_size
        self.tokenizer = tokenizer
        self.context_token = '[CONTEXT]'
        self.response_token = '[RESPONSE]'
        self.context_token_id = self.tokenizer.convert_tokens_to_ids(self.context_token)
        self.response_token_id = self.tokenizer.convert_tokens_to_ids(self.response_token)
        #self.special_ids = special_ids
        self.debug = debug

        self.dataset_map = {}
        self.dataset_paths = {}
        if use_dailydialog is True:
            self.dataset_map['DialyDialog'] = Parsers.get_dailydialog_dialogues
            self.dataset_paths['DialyDialog'] = './data/model_training/ijcnlp_dailydialog/'
        if use_personachat is True:
            self.dataset_map['Personachat'] = Parsers.get_personachat_dataset_dialogues
            self.dataset_paths['Personachat'] = './data/model_training/personachat_self_original.json'
        if use_taskmaster is True:
            self.dataset_map['Taskmaster'] = Parsers.get_taskmaster_dialogues
            self.dataset_paths['Taskmaster'] = './data/model_training/Taskmaster/'

        self.dialogues, self.topics, self.emotions_map = self._load_dialogues_()
        random.seed(seed)
        random.shuffle(self.dialogues)
        
        if self.mode is not None:
            train_len = math.ceil(len(self.dialogues) * self.train_size)
            if self.mode == 'train':
                self.dialogues = self.dialogues[:train_len]
            else:
                self.dialogues = self.dialogues[train_len:]

        self.topics_map = {topic: num for num, topic in enumerate(self.topics)}

        if use_sequence_bucketing is True:
            self.dialogues = self._bucket_sequences_(self.dialogues)

    def _bucket_sequences_(self, data):
        return data

    def get_topics(self):
        return self.topics

    def _load_dialogues_(self):
        dialogues, topics = [], []
        for dataset_name in list(self.dataset_map.keys()):
            dialogue, second_param = self.dataset_map[dataset_name](
                dataset_path=self.dataset_paths[dataset_name],
                debug=self.debug,
                use_emotions=self.use_dailydialog_emotions
            )
            emotions_map = None
            if self.use_dailydialog_emotions is True:
                topic, emotions_map = second_param
            else:
                topic = second_param
            splitted_dialogues = []
            for d in dialogue:
                splitted_dialogues.extend(d.split())
            dialogues.extend(splitted_dialogues)
            topics.extend(topic)
        return dialogues, topics, emotions_map


    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        context = dialogue['context']
        context = ' '.join([ctx[0] + ' ' + ctx[1] for ctx in context])
        response = dialogue['response']
        response = response[0] + ' ' + response[1]
        topic = dialogue['topic']
        
        context = self.tokenizer.tokenize(context)
        response = self.tokenizer.tokenize(response)

        output = {}
        if self.model_type == 'gpt2':
            context = context[-(self.context_max_len-1):]
            response = response[:(self.response_max_len-2)]
            context = [self.context_token] + context
            response = [self.response_token] + response + [self.tokenizer.eos_token]
            context_tt_ids = [self.context_token_id] * len(context)
            response_tt_ids = [self.response_token_id] * len(response)
            
            output['real_context_len'] = len(context)
            output['context_tt_ids'] = context_tt_ids
            output['response_tt_ids'] = response_tt_ids
            #output['labels'] = context
        else: # for BART model
            context = context[-(self.context_max_len-2):]
            response = response[:(self.response_max_len-2)]
            context = [self.context_token] + context + [self.tokenizer.eos_token]
            response = [self.response_token] + response + [self.tokenizer.eos_token]

        context = self.tokenizer.convert_tokens_to_ids(context)
        response = self.tokenizer.convert_tokens_to_ids(response)

        output['context_ids'] = context
        output['response_ids'] = response

        if self.use_topic is True:
            output['label'] = self.topics_map[topic]
        
        if self.use_dailydialog_emotions is True:
            output['label'] = dialogue['emotions']

        return output


def collate_fn(batch_data : List, pad_elem: int, num_neg_samples: int = 3, model_type: str = 'gpt2'):
    def pad(seq: list, max_len: int, pad_elem: int) -> List:
        return seq + [pad_elem] * max(0, max_len - len(seq))
    
    output = {}
    batch_size = len(batch_data)
    num_neg_samples = min(num_neg_samples, batch_size - 1)
    
    # assemble data: list[dict] -> dict{list}
    keys = batch_data[0].keys()
    samples = [[sample[key] for key in keys] 
               for sample in batch_data]
    batch_data = {key: data_part
                  for key, data_part
                  in zip(keys, map(list, zip(*samples)))}
    
    context_ids = batch_data['context_ids']
    response_ids = batch_data['response_ids']
    if model_type == 'gpt2':
        batch_ids = []
        batch_tt_ids = []
        context_tt_ids = batch_data['context_tt_ids']
        response_tt_ids = batch_data['response_tt_ids']

        # it gives (cont1, resp1), (cont1, random_resp1), (cont1, random_resp2),... 
        #          (cont2, resp2), (cont2, random_resp1), etc.
        for pos_indx in range(batch_size):
            _context = context_ids[pos_indx]
            _context_tt = context_tt_ids[pos_indx]
            
            # create negative indices for response_ids
            neg_indices = random.sample(range(batch_size), num_neg_samples)
            # don't use positive sample as negative
            neg_indices = [i if i != pos_indx
                           else (i + 1) % batch_size
                           for i in neg_indices]

            # add positive sample with positive indx then add negative samples to the batch
            for indx in [pos_indx, *neg_indices]:
                sample_ids = _context + response_ids[indx]
                sample_tt_ids = _context_tt + response_tt_ids[indx]
                batch_ids.append(sample_ids)
                batch_tt_ids.append(sample_tt_ids)
            
        max_seq_len = max(map(len, batch_ids))
        
        context_ids, context_tt_ids = [
            torch.LongTensor(
                [pad(sample, max_seq_len, pad_elem) for sample in arr]
            )
            for arr in (batch_ids, batch_tt_ids)
        ]
        
        context_pos_ids = torch.arange(0, max_seq_len).repeat(context_ids.size(0), 1)
        attn_mask = context_ids.ne(pad_elem).long()
        # repeat real_context_len: len_1, ..., len_1, ..., len_batch_size, ..., len_batch_size
        real_context_len = torch.LongTensor(batch_data['real_context_len']).unsqueeze(1)
        real_context_len = real_context_len.repeat(1, 1 + num_neg_samples).view(-1)

        output['context_ids'] = context_ids
        output['context_pos_ids'] = context_pos_ids
        output['context_tt_ids'] = context_tt_ids
        output['real_context_len'] = real_context_len
        output['attention_mask'] = attn_mask
        
        assert attn_mask.size() == context_tt_ids.size() == context_ids.size() == context_pos_ids.size()
    else:
        context_max_len = max(map(len, context_ids))
        response_max_len = max(map(len, response_ids))
        
        context_ids = torch.LongTensor(
                [pad(sample, context_max_len, pad_elem) for sample in context_ids]
        )
        context_ids = context_ids.repeat(1, 1 + num_neg_samples).view(-1, context_max_len)
        # gives [context_#1, ..., context_#1, ..., context_#batch_size, ..., context#batch_size] 
        # where each context_#num is repeated num_neg_samples times
        # context_ids.size() = torch.Size([batch_size*num_neg_samples, context_max_len])
        context_attn_mask = context_ids.ne(pad_elem).long()

        response_ids = torch.LongTensor(
                [pad(sample, response_max_len, pad_elem) for sample in response_ids]
        )
        response_permutation = []
        for pos_indx in range(batch_size):
            response_permutation.append(pos_indx)
            neg_indices = random.sample(range(batch_size), num_neg_samples)
            # don't use positive sample as negative
            neg_indices = [i if i != pos_indx
                           else (i + 1) % batch_size
                           for i in neg_indices]
            response_permutation.extend(neg_indices)
        response_ids = response_ids[response_permutation]
        # it gives [response_#1, then negative responses, response#2, then negative responses, etc.
        # len(negative responses) = num_neg_samples
        # each negative response from negative responses is a random response from the batch 
        # negative response != current response in a mini batch
        # len(mini_batch) = num_neg_samples + 1
        # response_ids.size() = torch.Size([batch_size*num_neg_samples, response_max_len])
        response_attn_mask = response_ids.ne(pad_elem).long()

        output['context_ids'] = context_ids
        output['attention_mask'] = context_attn_mask
        output['response_ids'] = response_ids
        output['decoder_attention_mask'] = response_attn_mask

        assert context_ids.size(0) == response_ids.size(0) == context_attn_mask.size(0) == response_attn_mask.size(0)

    # these are indices for language modeling loss, we want to compute lm_loss only on positive samples
    # positive sample always at the beginning of a mini batch in the pairs (context, response)
    # len(mini batch) = num_neg_samples + 1
    lm_indices = torch.arange(0, batch_size*(1 + num_neg_samples), num_neg_samples + 1)

    # these are labels for classification loss, positive samples always at the beginning of a mini batch
    mc_labels = torch.LongTensor(
        [1 if not idx % batch_size else 0 for idx in range(context_ids.size(0))]
    )

    output['lm_indices'] = lm_indices
    output['cls_head_labels'] = mc_labels

    if 'label' in batch_data:
        output['label'] = torch.LongTensor(batch_data['label'])

    return output

# TODO: add support of AttrDataset for non-dialog data
def get_dataloaders(tokenizer, model_type, batch_size, num_neg_samples=0, num_workers=0, return_topic_list=False, **kwargs):
    train_dataset = DialogueDataset(tokenizer, model_type, mode='train', **kwargs)
    val_dataset = DialogueDataset(tokenizer, model_type, mode='valid', **kwargs)
    # define model_type for the collate function
    pad_tok_id = tokenizer.eos_token_id if model_type == 'gpt2' else tokenizer.pad_token_id
    from .dataset import collate_fn
    collate_fn = partial(collate_fn, pad_elem=pad_tok_id, num_neg_samples=num_neg_samples, model_type=model_type)
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, collate_fn=collate_fn, num_workers=num_workers)
    if return_topic_list:
        topics = train_dataset.get_topics()
        output = [train_loader, val_loader, topics]
    else:
        output = [train_loader, val_loader]
    return output
    
