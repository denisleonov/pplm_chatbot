import torch

from pytorch_lightning.metrics import Metric, TensorMetric
from pytorch_lightning.metrics.nlp import BLEUScore
from rouge import Rouge
from torch.nn import functional as F


class Perplexity(Metric):
    """
    Computes the perplexity of the model.
    """

    def __init__(self, pad_idx: int, *args, **kwargs):
        super().__init__(name='ppl')
        self.pad_idx = pad_idx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(pred, target, reduction='none')
        non_padding = target.ne(self.pad_idx)
        loss = loss.masked_select(non_padding).sum()

        num_words = non_padding.sum()
        ppl = torch.exp(
            torch.min(loss / num_words, torch.tensor([100]).type_as(loss))
        )
        return ppl


class RougeMetric(Metric):
    """
    Computes rouge metric
    """
    def __init__(self, *args, **kwargs):
        super().__init__(name='rouge')
        self.rouge = Rouge()

    def forward(self, pred_responses: list, target_responses: list) -> list:
        """Return rouge scores of responces.
        :return list of ROUGE (1,2,l) and mean of this scores

        ROUGE scores are computed for every pair of predicted and gold sentences. 
        We use only F-measure, that means "weighted measure" of precision and recall in terms of n-gramms.
        """
        scores = self.rouge.get_scores(pred_responses, target_responses, avg=True)
        scores = {
            f'rouge-{x}': torch.tensor([scores[f'rouge-{x}']['f']]) 
            for x in ('1', '2', 'l')
        }
        scores['rouge-mean'] = sum(scores.values()) / 3

        return scores
