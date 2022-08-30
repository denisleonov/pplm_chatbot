import random
import re

import numpy as np
from unidecode import unidecode

SQUEEZE_REPEATED = [r"\'\`\!\?\."]

class TextTransforms:

    @staticmethod
    def standardize(text):

        def sent_standardize(sent):
            sent = unidecode(sent)
            sent = re.sub(rf"([{SQUEEZE_REPEATED}])(?=[{SQUEEZE_REPEATED}]*\1)", "", sent)
            sent = re.sub(' +', ' ', sent)
            return sent.strip()

        return [sent_standardize(sent) for sent in text]

    @staticmethod
    def random_rcut(dialog):
        cut = random.randrange(2, len(dialog), 2)
        return dialog[:cut]

    @staticmethod
    def random_lcut(dialog):
        cut = random.randrange(0, len(dialog) - 1, 2)
        return dialog[cut:]

    @staticmethod
    def shuffle_and_cut(persona_info, min_info_sent=3):
        n_info_samples = np.random.randint(min_info_sent, len(persona_info) + 1)
        persona_info = random.sample(persona_info, k=n_info_samples)
        random.shuffle(persona_info)
        return persona_info
