import random
import torch
import torch.nn.functional as F

import json

from src.dataset_parsers import Parsers, Dialogue
from collections import deque
from itertools import chain
from colorama import init, Fore, Back, Style

class DialogHistory():
    def __init__(self, context_spec_token, model_type='bart', size=128, device='cuda'):
        self.dialog_ids = []
        self.dialog_tt_ids = []
        
        self.device = device
        # num of tokens to keep in the history (-1 for [CONTEXT] token)
        self.dialog_size = size - 1
        self.context_spec_token = context_spec_token
        self.model_type = model_type


    def append(self, spec_token_id, sentence_ids, context_spec_token=None):
        '''
        Append utterance in the dialog.
        spec_token_id (int): special id in the vocabulary for the type of a speaker (other/ self)
        sentence_ids (list[int]): ids from tokenizer.decode method
        '''
        
        sentence_ids_ = [spec_token_id] + sentence_ids
        self.dialog_ids += sentence_ids_ # extend with new utterance
        self.dialog_ids = self.dialog_ids[-self.dialog_size:] # keep last #dialog_size tokens

    def get_inputs(self):
        '''
        Function for building inputs for the model
        '''
        ids = torch.tensor([self.context_spec_token] + self.dialog_ids).long().to(self.device)
        tt_ids = pos_ids = None
        if self.model_type == 'gpt2':
            dialog_tt_ids = [self.context_spec_token] * ids.size(-1)
            tt_ids = torch.tensor(dialog_tt_ids).long().to(self.device)
            pos_ids = torch.arange(ids.size(-1)).long().to(self.device)

        return ids, tt_ids, pos_ids

class ChatBot():
    def __init__(self, model, device, generator, history_size):
        init()
        self.model = model
        self.device = device
        self.generator = generator
        self.colored = False
        self.history_size = history_size

        self.history = DialogHistory(
            context_spec_token=model.special_ids.context,
            model_type=model.model_type,
            size=self.history_size,
            device=self.device
        )   
    
    def interact(self, pplm=False, colored=False):
        self.colored = colored

        utterance = ''
        while utterance.lower() != 'exit':
            utterance = input('>>> ')
            while not utterance:
                print('Prompt should not be empty!')
                utterance = input('>>> ')
            if pplm:
                reply = self.reply(utterance, pplm)
            else:
                with torch.no_grad():
                    reply = self.reply(utterance, pplm)
            print('bot: ', reply)
        else:
            exit()

    def reply(self, raw_text, pplm=False):
        in_ids = self.model.tokenizer.encode(raw_text, add_special_tokens=False)
        self.history.append(
            self.model.special_ids.alice,
            in_ids
        )
        inputs = self.history.get_inputs()
        #print(inputs[0])
        #print('DBG: ', self.model.tokenizer.convert_ids_to_tokens(inputs[0], skip_special_tokens=False))
        self.generator.initialize(*inputs)
        out_ids = self.generator.run().detach().cpu().tolist()

        self.history.append(
            self.model.special_ids.bob,
            out_ids
        )
        
        # print colored words from the BoWs if needed
        if pplm and self.generator.attr_model.ohv and self.colored:
            reply = ''
            for word_id in out_ids:
                word = self.model.tokenizer.decode([word_id], skip_special_tokens=False)
                if word_id in self.generator.attr_model.bow_indices:
                    reply += '{}{}{}'.format(
                        Fore.RED,
                        word,
                        Style.RESET_ALL
                    )
                else:
                    reply += word
        else:
            reply = self.model.tokenizer.decode(out_ids, skip_special_tokens=False)

        return reply
