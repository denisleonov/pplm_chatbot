import numpy as np
import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from operator import add
from typing import List
from functools import partial

from src.utils.bow import get_bag_of_words_indices, build_bows_one_hot_vectors
from src.utils.ngram_elim import BatchNgramEliminator, DummyNgramEliminator
from src.utils.utils import add_to_sequence, apply_threshold


class AbstractDecodeStrategy:
    """ Abstract class for decode strategy """
    current_pos: int
    responce_len: int
    ids: torch.LongTensor
    tt_ids: torch.LongTensor
    mask_ids: torch.LongTensor
    pos_ids: torch.LongTensor
    ngram_eliminator: BatchNgramEliminator

    def __init__(self, model, spec_ids, device, max_responce_len, amp_enabled=False, ngram_len=3):
        self.model = model
        self.using_gpt2 = model.model_type == 'gpt2'
        self.spec_ids = spec_ids
        self.device = device
        self.max_responce_len = max_responce_len
        self.amp_enabled = amp_enabled
        self.block_ngrams = ngram_len not in [None, 0]
        self.ngram_len = ngram_len

    def initialize(self, *args, **kwargs):
        raise NotImplementedError(f"You must realize this in {self.__class__.__name__}")

    def run(self, *args, **kwargs):
        raise NotImplementedError(f"You must realize this in {self.__class__.__name__}")


class SamplingStrategy(AbstractDecodeStrategy):
    """ Abstract class for sampling-based strategies: greedy, top-k, top-p"""
    isend: bool

    def initialize(self, context_ids, context_tt_ids=None, context_pos_ids=None, **extra):

        self.current_pos = context_ids.size(-1)
        self.responce_len = 1

        # add SOS token | torch.Size([1, prefix_len])
        self.context_ids = context_ids.unsqueeze(0)
        self.context_tt_ids = context_tt_ids.unsqueeze(0) if context_tt_ids is not None else None
        self.context_pos_ids = context_pos_ids.unsqueeze(0) if context_pos_ids is not None else None
        self.response_ids = torch.tensor([[self.spec_ids.response]]).type_as(self.context_ids)
        
        *_, self.encoder_outputs = self.model.get_hidden_and_past(
            context_ids=self.context_ids,
            context_tt_ids=self.context_tt_ids,
            context_pos_ids=self.context_pos_ids,
        )

        # initialize trigram eliminator
        self.ngram_eliminator = BatchNgramEliminator(
            size=1, device=self.device, ngram_len=self.ngram_len, amp_enabled=self.amp_enabled
        ) if self.block_ngrams else DummyNgramEliminator()

        # is end
        self.isend = False

    def run(self):
        while self.responce_len <= self.max_responce_len and not self.isend:
            if self.using_gpt2:
                self.context_ids, self.context_tt_ids, self.context_pos_ids = [
                    add_to_sequence(prefix, elem)
                    for prefix, elem
                    in zip(
                        [self.context_ids, self.context_tt_ids, self.context_pos_ids],
                        [self.response_ids[0, -1], self.spec_ids.response, self.current_pos],
                    )
                ]

            probs = self.get_model_probs()

            # sampling | int
            sample_id = self.sample(probs).item()
            self.isend = sample_id in (self.model.tokenizer.eos_token_id, self.spec_ids.pad)
            if self.isend:
                break
            self.responce_len += 1
            self.current_pos += 1

            # block n-gram
            # self.ngram_eliminator.refresh(sample_id)

            # add new token to the sequence | torch.Size([1, current_pos])
            self.response_ids = add_to_sequence(self.response_ids, sample_id)

        # pull responce ids without [RESPONSE] token | torch.Size([responce_len - 1])
        return self.response_ids[0, 1:]

    def sample(self, *args, **kwargs):
        raise NotImplementedError(f"You must realize this in {self.__class__.__name__}")

    def get_model_probs(self):            
        # get last hidden of the model at t-th time step | torch.Size([1, len, d_hidden])
        hidden, *_ = self.model.get_hidden_and_past(
            context_ids=self.context_ids,
            context_tt_ids=self.context_tt_ids,
            context_pos_ids=self.context_pos_ids,
            response_ids=self.response_ids,
            encoder_outputs=self.encoder_outputs
        )

        # get output for last token in the input sequence | torch.Size([1, d_hidden])
        hidden = hidden[:, -1, :]

        # return vocab probs for the next token | torch.Size([1, vocab_size])
        return self.model.lm_head.get_probs(hidden)


class GreedySampling(SamplingStrategy):

    def sample(self, out):
        """ Greedily choose ids by given decoder out """

        # torch.Size([1, vocab_size])
        logprobs = self.model.lm_head(out)
        # torch.Size([1, vocab_size])
        logprobs = self.ngram_eliminator.eliminate_idxs(logprobs)
        # get best id | torch.Size([1])
        sample_id = logprobs.argmax(-1)

        return sample_id


class TopPSampling(SamplingStrategy):
    """ Implement top-p (nucleus) sampling """

    def __init__(self, *args, **kwargs) -> None:
        assert 'top_p' in kwargs
        self.top_p = kwargs.pop('top_p')
        super().__init__(*args, **kwargs)

    def sample(self, probs):
        """ Sample ids by given decoder out according to top-p strategy """

        # torch.Size([1, vocab_size])
        # probs = self.ngram_eliminator.eliminate_idxs(probs, is_probs=True)
        # torch.Size([1, vocab_size])
        sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
        # torch.Size([1, vocab_size])
        cumulative_probs = torch.cumsum(sorted_probs, dim=1)
        # torch.Size([1, vocab_size])
        sorted_probs[cumulative_probs > self.top_p] = 0.
        sorted_probs[sorted_probs[:, 0] == 0, 0] = 1.
        # torch.Size([1, vocab_size])
        sorted_probs /= sorted_probs.sum(dim=1, keepdim=True)
        # torch.Size([1])
        sample = Categorical(sorted_probs).sample()
        sample_id = sorted_idxs.gather(1, sample.unsqueeze(1)).squeeze(1)

        return sample_id
