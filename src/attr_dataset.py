import math

import torch


class AttrDataset(torch.utils.data.Dataset):
   
    def __init__(self,
                 data,
                 tokenizer,
                 train_size=0.75,
                 ctx_max_len=256,
                 rsp_max_len=100,
                 seed=42,
                 split_to_ctx_rsp=False,
                 mode='train',
                 model_type='gpt2'
                ):
        assert model_type in ['gpt2', 'bart']
        assert mode in ['train', 'valid', None]
        assert 'additional_special_tokens' in tokenizer.special_tokens_map, 'Add special tokens before usage.'
        self.model_type = model_type
        self.data = data
        self.tokenizer = tokenizer
        self.train_size = train_size
        self.ctx_max_len = ctx_max_len
        self.rsp_max_len = rsp_max_len
        self.seed = seed
        self.mode = mode
        self.split_to_ctx_rsp = split_to_ctx_rsp
        self.context_token = '[CONTEXT]'
        self.response_token = '[RESPONSE]'
        self.context_token_id = self.tokenizer.convert_tokens_to_ids(self.context_token)
        self.response_token_id = self.tokenizer.convert_tokens_to_ids(self.response_token)

        if self.mode is not None:
            train_len = math.ceil(len(self.data) * self.train_size)
            if self.mode == 'train':
                self.data = self.data[:train_len]
            else:
                self.data = self.data[train_len:]

    def __getitem__(self, idx):
        #sample = self.data[idx]
        pass