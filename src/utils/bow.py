import os
import torch
from pathlib import Path

import warnings

def get_bag_of_words_indices(bag_of_words_paths, tokenizer, path_to_bow=Path('data/bow')):
    bow_indices = []
    for filename in bag_of_words_paths:
        filepath = path_to_bow / Path(filename + '.txt')
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                words = f.read().strip().split("\n")
            bow_indices.append(
                [tokenizer.encode(word.strip(),
                                  add_prefix_space=True,
                                  add_special_tokens=False)
                 for word in words])
        else:
            warnings.warn('Cannot add \'' + filename + '\': file with words do not exist. Skipping...')
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, len(tokenizer)).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors