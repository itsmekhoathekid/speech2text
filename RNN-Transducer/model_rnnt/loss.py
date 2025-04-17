
import torch
import torch.nn as nn

class TransducerLoss(nn.Module):
    r"""
    Compute path-aware regularization transducer loss.

    Args:
        configs (DictConfig): hydra configuration set
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        logits (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        targets (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        input_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        target_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.

    Returns:
        - loss (torch.FloatTensor): transducer loss

    Reference:
        A. Graves: Sequence Transduction with Recurrent Neural Networks:
        https://arxiv.org/abs/1211.3711.pdf
    """

    def __init__(self, blank_id, reduction, gather) -> None:
        super().__init__()
        from warp_rnnt import rnnt_loss
        self.rnnt_loss = rnnt_loss
        self.blank_id = blank_id
        self.reduction = reduction
        self.gather = gather

    def forward(
            self,
            logits: torch.FloatTensor,
            targets: torch.IntTensor,
            input_lengths: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        return self.rnnt_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
            reduction=self.reduction,
            blank=self.blank_id,
            gather=self.gather,
        )