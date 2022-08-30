import copy
import torch
from collections import defaultdict, deque


class SingleSeqNgramEliminator:
    """ Control N-gram repeatedness for single sequence """

    def __init__(self, ngram_len):
        self.ngram_len = ngram_len
        self.prefix2idxs = defaultdict(list)
        self.last_ngram = deque(maxlen=ngram_len)

    def refresh(self, idx):
        """ Update last N-gram for another idx"""
        self.last_ngram.append(idx)
        if len(self.last_ngram) == self.ngram_len:
            *prefix, idx = self.last_ngram
            self.prefix2idxs[tuple(prefix)].append(idx)

    def eliminate_idxs(self):
        """ Determine idxs to eliminate for last prefix"""
        if len(self.last_ngram) == self.ngram_len:
            _, *prefix = self.last_ngram
            return self.prefix2idxs.get(tuple(prefix), [])
        return []


class BatchNgramEliminator:
    """ Control N-gram repeatedness for all sequences in batch """

    def __init__(self, size, device, ngram_len, amp_enabled, beam_size=1):
        self.buffer = [SingleSeqNgramEliminator(ngram_len) for _ in range(size)]
        self.device = device
        self.amp_enabled = amp_enabled
        self.beam_size = beam_size

    def refresh(self, token_idxs):
        """ Update last N-grams in all sequences """
        token_idxs = token_idxs.tolist() if isinstance(token_idxs, torch.Tensor) else token_idxs
        for ngram_elim, tkn_idx in zip(self.buffer, token_idxs):
            ngram_elim.refresh(tkn_idx)

    def restruct(self, channels):
        """ Change buffers order for surviving sequences after beam step """
        self.buffer = [
            copy.deepcopy(self.buffer[self.beam_size * (idx // self.beam_size) + ch])
            for idx, ch in enumerate(channels.tolist())
        ]

    def eliminate_idxs(self, batch_logprobs, is_probs=False):
        """ Nullify logprobs according to N-gram Eliminators in batch """
        for NgramElim, seq_logprobs in zip(self.buffer, batch_logprobs):
            ids_to_eliminate = torch.tensor(NgramElim.eliminate_idxs(), dtype=torch.long, device=self.device)
            seq_logprobs[ids_to_eliminate] = 0. if is_probs else (-1e4 if self.amp_enabled else -1e20)
        return batch_logprobs


class DummyNgramEliminator:

    def __init__(self):
        pass

    def refresh(self, token_idxs):
        pass

    def restruct(self, channels):
        pass

    def eliminate_idxs(self, batch_logprobs, *args, **kwargs):
        return batch_logprobs
