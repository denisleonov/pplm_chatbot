import torch
import torch.nn as nn


class UnlikelihoodLoss(nn.Module):
    """
    Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
    """

    def __init__(self, pad_idx, alpha=0.25):
        super().__init__()
        self.pad_idx = pad_idx
        self.alpha = alpha

    def forward(self, logprobs, target):
        """
        pred: logprobs |  torch.Size([n_examples, vocab_size])
        target: target ids | torch.Size([n_examples])
        """
        # N, T -> N * T
        target = target.view(-1)
        # N, T, V -> N * T, V
        logprobs = logprobs.view(-1, logprobs.size(-1))

        # Form negative targets:
        # e.g. DABBC | D | EFFGD => {A,B,C} are negative targets,
        # where | - SEP token. 
        # Thus, we prevent the generation of tokens that have already been generated.
        with torch.no_grad():
            # for each sentence and each timestep initialize negative targets as targets
            # equals [[target] for n in range(len(targets))]
            neg_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
            # neg_cands for first generation: None;
            # for second generation: first token in the targets;
            # for third: first, second token; etc.
            # thus, mask = upper triangular matrix
            mask = torch.empty_like(neg_cands).fill_(self.pad_idx).triu()
            neg_cands = neg_cands.tril(-1) + mask

            # don't include the target for that timestep as a negative target
            neg_cands = neg_cands.masked_fill(neg_cands == target.unsqueeze(1), self.pad_idx)
            negative_targets = torch.zeros_like(logprobs).scatter_(1, neg_cands, 1)            

        # loss
        inverse_probs = torch.clamp((1.0 - logprobs.exp()), min=1e-5)
        ul_loss = -torch.log(inverse_probs) * negative_targets

        return self.alpha * ul_loss.sum()

class SmoothLoss(nn.Module):
    """
    Implement label smoothing.
    """

    def __init__(self, smooth_eps, vocab_size, pad_idx):
        self.vocab_size = vocab_size
        self.smooth_eps = smooth_eps
        self.pad_idx = pad_idx
        super().__init__()

    def forward(self, pred, target):
        """
        :param pred: logprobs |  torch.Size([n_examples, vocab_size])
        :param target: target ids | torch.Size([n_examples])
        :return:
        """

        # one hot encoding | torch.Size([n_examples, vocab_size])
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        # fill with smoothed labels | torch.Size([n_examples, vocab_size])
        one_hot = one_hot * (1 - self.smooth_eps) + \
            torch.sub(1, one_hot) * self.smooth_eps / (self.vocab_size - 1)
        # sum over vocab + masking | torch.Size([n_examples])
        loss = - (one_hot * pred).sum(dim=1)
        # masking padding ids
        loss = loss.masked_select(target.ne(self.pad_idx)).mean()
        return loss


if __name__ == '__main__':
    """
    Some test's
    """

    eps = 0.1
    n_exs = 10
    vcb_sz = 5
    pad_id = 0
    criter = SmoothLoss(eps, vcb_sz, pad_id)

    p = torch.randn(n_exs, vcb_sz)
    t = torch.empty(n_exs).random_(0, vcb_sz).long()

    print("pred:\n", p)
    print("target:\n", t)

    print(criter(p, t))
