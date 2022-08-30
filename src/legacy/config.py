from pathlib import Path

# model
MIN_INFO_SENTS = 3
MAX_LEN = 512

# additional
NUM_WORKERS = 0
SEED = 1234

# paths
# use data from PeronaChat competition
PATH2DATA = Path('data/personachat')
DS_FILES = [
    (PATH2DATA / f'{split}_both_original.txt', PATH2DATA / f'{split}_both_revised.txt')
    for split in ['train', 'valid', 'test']
    ]

#initial configuration for heads
PATH2CONFIG = 'head_configs'

#initial set of heads and bow-vectors - here could be all available heads
HEADS = ['topics_extended', 'gender', 'decade'] #[ 'subject','toxic', 'sentiment']
TOPICS = ['space']

PATH2WEIGHTS = Path('weights')
PATH2BOW = Path('bow')

if __name__ == "__main__":
    print(DS_FILES)
