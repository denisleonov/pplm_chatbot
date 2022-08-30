import torch
import torch.nn as nn

from collections import namedtuple
from pathlib import Path
from transformers import AutoTokenizer, BartModel, GPT2LMHeadModel
from torch.nn import functional as F

import omegaconf

AVAILABLE_MODELS = [
    'facebook/bart-base',
    'facebook/bart-large',
    'microsoft/DialoGPT-small',
    'microsoft/DialoGPT-medium',
    'microsoft/DialoGPT-large',
    'gpt2',
    'gpt2-medium',
    'gpt2-large'
]
SPECIAL_TOKENS = ['CONTEXT', 'RESPONSE', 'ALICE', 'BOB']
SPECIAL_TOKENS_GPT = SPECIAL_TOKENS + ['pad']
SPECIAL_TOKENS_BART = SPECIAL_TOKENS + ['pad', 'exptok']
SpecialIdsGPT = namedtuple('special_ids', [t.lower() for t in SPECIAL_TOKENS_GPT])
SpecialIdsBART = namedtuple('special_ids', [t.lower() for t in SPECIAL_TOKENS_BART])


class Generator(nn.Module):
    """Generator without perturbation"""
    def __init__(self, d_hidden, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_hidden, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, is_logits=False):
        """Returns log probabilities of tokens"""
        logits = x if is_logits else self.proj(x)
        return self.log_softmax(logits)

    def get_probs(self, x, is_logits=False, temperature=1.0):
        """Returns probabilities of tokens using softmax"""
        logits = x if is_logits else self.proj(x)
        return self.softmax(logits / temperature)

class SequenceSummary(nn.Module):
    """Compute a single vector summary of a sequence hidden states."""

    def __init__(self, d_hidden, pad_id, n_classes=2, first_dropout=0.1, last_dropout=0.1):
        super().__init__()
        
        self.pad_id = pad_id
        self.first_dropout = nn.Dropout(first_dropout)
        self.summary = nn.Linear(d_hidden, n_classes)
        self.last_dropout = nn.Dropout(last_dropout)

    def forward(self, hidden_states, input_ids):
        """Compute a single vector summary of a sequence hidden states."""
        nonpad_mask = input_ids.ne(self.pad_id).float()
        nonpad_mask = nonpad_mask.unsqueeze(-1)

        x = torch.sum(hidden_states * nonpad_mask, 1) / torch.sum(nonpad_mask, 1)
        x = self.first_dropout(x)
        x = F.gelu(self.summary(x))
        x = self.last_dropout(x)

        return x


class DoubleHeadsModel(nn.Module):
    """
    Class to wrap GPT-2 model.
    """
    def __init__(self, 
                 model_name, 
                 tokenizer, 
                 special_ids, 
                 use_cls_head=False, 
                 path_to_weights=Path('weights'), 
                 train_transformer=False, 
                 use_cache=False):
        super().__init__()
        
        self.train_transformer = train_transformer

        self.special_ids = special_ids
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.using_cache = use_cache

        self.path_to_weights = path_to_weights

        assert model_name.startswith(('facebook/bart', 'gpt2', 'microsoft/DialoGPT'))
        self.model_name = model_name
        self.transformer = self.build_transformer(self.vocab_size, model_name)
        if model_name.startswith('facebook/bart'):
            self.model_type = 'bart'
        elif model_name.startswith(('gpt2', 'microsoft/DialoGPT')):
            self.model_type = 'gpt2'
        
        #self.decoder = lambda *args, **kwargs: self.transformer(*args, **kwargs)[0]
        self.hidden_size = self.transformer.config.hidden_size
        
        if use_cls_head:
            self.classification_head = SequenceSummary(self.hidden_size, self.special_ids.pad)
        
        self.lm_head = Generator(self.hidden_size, self.vocab_size)
        if hasattr(self.transformer, 'lm_head'):
            # then transformer is transformer + lm_head, but we use own wrapper
            # reusing lm_head weights
            self.lm_head.proj.weight.data = self.transformer.lm_head.weight.data
            del self.transformer.lm_head
            self.transformer = self.transformer.transformer
        
        self.past_transforms = {
            'bart': {'totensor': self._past_totensor_bart,
                                   'fromtensor': self._past_fromtensor_bart},
            'gpt2': {'totensor': self._past_totensor_gpt2,
                     'fromtensor': self._past_fromtensor_gpt2}
        }

    def decoder(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)[0]

    def use_cache(self, mode):
        self.using_cache = mode

    def past_totensor(self, past):
        return self.past_transforms[self.model_type]['totensor'](past)

    def past_fromtensor(self, past):
        return self.past_transforms[self.model_type]['fromtensor'](past)

    def save(self, save_filename):
        """Saves model state dict state dicts of parts at given path.
        
        :param:
            save_filename: name of file with weights (without path)
        """
        state = {
            'transformer_dict': self.transformer.state_dict(),
            'lm_head_dict': self.lm_head.state_dict(),
            'tokenizer': self.tokenizer,
        }
        model_path = self.path_to_weights / Path(save_filename.name + '_' + self.model_name.replace('/', '_'))
        torch.save(state, model_path)

    def load(self, load_filename):
        """Loads model state dict with state dicts of model parts

        :param:
            load_filename: name of file with weights (without path)
        """
        print('===> Loading model weights')
        base_path = self.path_to_weights / load_filename
        print(base_path)
        state = torch.load(base_path)
        self.transformer.load_state_dict(state['transformer_dict']) # transformer_dict
        self.lm_head.load_state_dict(state['lm_head_dict']) # lm_head_dict
        self.tokenizer = state['tokenizer']
        print('loaded base model weights')

    def forward(self,
                context_ids,
                response_ids=None,
                context_tt_ids=None,
                context_pos_ids=None,
                attention_mask=None,
                decoder_attention_mask=None,
                bart_enc_outputs=None,
                **extra):
        last_hidden, *_ = self.get_hidden_and_past(
            context_ids=context_ids,
            context_tt_ids=context_tt_ids,
            context_pos_ids=context_pos_ids,
            response_ids=response_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=bart_enc_outputs
        )
        logprobs = self.lm_head(last_hidden)

        cls_logits = None
        if response_ids is not None:
            input_ids = response_ids
        else:
            input_ids = context_ids

        if hasattr(self, 'classification_head'):
            cls_logits = self.classification_head(last_hidden, input_ids)
        
        return logprobs, cls_logits
    
    def get_hidden_and_past(self,
                            inputs_embeds=None,
                            context_ids=None,
                            context_tt_ids=None,
                            context_pos_ids=None,
                            response_ids=None,
                            attention_mask=None,
                            decoder_attention_mask=None,
                            past=None,
                            encoder_outputs=None,
                            **extra):
        assert context_ids is None or inputs_embeds is None, \
            "You cannot specify both context_ids and inputs_embeds at the same time :("

        if isinstance(past, torch.Tensor):
            past = self.past_fromtensor(past)
        
        if self.model_type == 'gpt2':
            if inputs_embeds is None and past is not None:
                # If past is used, 
                # only input_ids that do not have their past calculated 
                # should be passed as input_ids.
                context_ids = context_ids[:, -1:]
                context_tt_ids = context_tt_ids[:, -1:]
                context_pos_ids = context_pos_ids[:, -1:]
                attention_mask = attention_mask[:, -1:]

            outputs =  self.transformer(
                input_ids=context_ids,
                token_type_ids=context_tt_ids,
                position_ids=context_pos_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past=past,
                use_cache=self.using_cache
            )
            if self.using_cache:
                hidden, past = outputs
                past = self.past_totensor(past)
            else:
                hidden = outputs[0]
                past = None

            return hidden, past, None
        elif self.model_type == 'bart':
            if inputs_embeds is not None:
                # use custom input_id for expected embeddings
                response_ids = torch.tensor([[self.special_ids.exptok]],
                                            dtype=torch.long)
                # get current model's embeddings
                wte = self.transformer.get_input_embeddings()
                # change embeddings of the expected token id directly
                wte.weight.data[self.special_ids.exptok] = inputs_embeds[0, 0, :]
                self.transformer.set_input_embeddings(wte)
            elif response_ids is None:
                # empty_ids is used for empty bart decoder input
                response_ids = torch.tensor([[]], dtype=torch.long, device=context_ids.device)
            
            # outputs: 
            # last_decoder_hidden_states, ((last_encoder_hidden_states, encoder_padding_mask), next_decoder_cache), last_encoder_hidden_states
            outputs = self.transformer(
                input_ids=context_ids,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=response_ids,
                decoder_cached_states=past,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=self.using_cache
            )
            
            hidden = outputs[0]
            if self.using_cache:
                encoder_outputs = outputs[1][0]
                past = outputs[1][1]
                past = self.past_totensor(past)
            else:
                past = encoder_outputs = None
                
            return hidden, past, encoder_outputs
        else:
            raise NotImplementedError(f'{self.model_name.upper()} is not supported yet')

    def _past_totensor_gpt2(self, past):
        # list -> tensor, new past shape:
        # (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head)
        return torch.stack(past)

    def _past_fromtensor_gpt2(self, past):
        # tensor -> list
        return list(map(lambda x: torch.squeeze(x, dim=0), past.split(1, dim=0)))

    def _past_totensor_bart(self, past):
        # need to remember for inverse transform
        self.enc_past_len = past[0]['encoder_decoder']['prev_key'].size(-2)
        
        # dict -> tensor, new past shape
        # (n_layers, 2, batch_size, num_heads, sequence_length, embed_size_per_head)
        past_ = []
        for idx, layer in enumerate(past):
            # concatenate keys along seq_len
            keys = torch.cat((layer['encoder_decoder']['prev_key'],
                              layer['self']['prev_key']), dim=-2)
            # concatenate values along seq_len
            values = torch.cat((layer['encoder_decoder']['prev_value'],
                                layer['self']['prev_value']), dim=-2)
            # stack keys and values in one piece and append in the past
            past_.append(
                torch.stack((keys, values))
            )
        return torch.stack(past_)

    def _past_fromtensor_bart(self, past):
        enclen = self.enc_past_len
        
        # tesor -> list[dict]
        past_ = []
        for idx in range(past.size(0)):
            layer_past = {'self': {}, 'encoder_decoder': {}}
            keys = past[idx, 0, ...]
            values = past[idx, 1, ...]
            
            layer_past['self']['prev_key'] = keys[..., enclen:, :] 
            layer_past['encoder_decoder']['prev_key'] = keys[..., :enclen, :] 
            
            layer_past['self']['prev_value'] = values[..., enclen:, :]
            layer_past['encoder_decoder']['prev_value'] = values[..., :enclen, :]

            past_.append(layer_past)
        return past_

    def eval(self):
        self.train(False)

    def train(self, mode=True):
        for param in self.transformer.parameters():
            param.requires_grad = self.train_transformer
        for param in self.lm_head.parameters():
            param.requires_grad = mode
        super().train(mode)

    @staticmethod
    def build_transformer(vocab_size, model_name):
        """Static method to build transformer model with given vocab size
        :param: vocab_size: size of the vocabluary
        :return: pretrained transformer
        """
        if model_name.startswith('facebook/bart'):
            transformer = BartModel.from_pretrained(model_name)
        elif model_name.startswith(('gpt2', 'microsoft/DialoGPT')):
            transformer = GPT2LMHeadModel.from_pretrained(model_name)
        transformer.resize_token_embeddings(vocab_size)
        
        return transformer
        
    @staticmethod
    def build_tokenizer(model_name):
        """Static method to build word tokenizer.
        
        Builds pretrained tokenizer and adds special tokens (pad etc)

        :return: instance of tokenizer and namedtuple with spetial ids
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name.startswith('facebook/bart'):
            special_dct = {t.lower(): f"[{t}]" for t in SPECIAL_TOKENS}
            special_dct['pad'] = tokenizer.pad_token
            special_dct['exptok'] = 'exptok'
            SpecialIds = SpecialIdsBART
        elif model_name.startswith(('gpt2', 'microsoft/DialoGPT')):
            special_dct = {t.lower(): f"[{t}]" for t in SPECIAL_TOKENS}
            special_dct['pad'] = tokenizer.eos_token
            SpecialIds = SpecialIdsGPT
        
        tokenizer.add_special_tokens({'additional_special_tokens': list(special_dct.values())})
        #special_ids = SpecialIds(**{k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in special_dct.items()})
        special_ids = omegaconf.OmegaConf.create({k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in special_dct.items()})
        return tokenizer, special_ids
