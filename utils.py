from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import numpy as np

class CosineWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = 1 * (epoch / self.warmup)
        else:
            lr_factor = 1
        return lr_factor

def batch_tokenize(seqs, alphabet, device="cpu"):
    """
    Tokenize a batch of RNA sequences using RNA-FM alphabet.
    
    seqs: list of strings (RNA sequences)
    alphabet: model.get_alphabet()
    device: "cpu" or "cuda"
    """
    # RNA-FM uses the same batch converter API as ESM
    batch_converter = alphabet.get_batch_converter()

    # Create dummy labels (ignored by model)
    labels_and_seqs = [(str(i), seq) for i, seq in enumerate(seqs)]

    _, _, tokens = batch_converter(labels_and_seqs)

    return tokens.to(device)
