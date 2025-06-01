from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from .encoder import TransformerTransducerEncoder
from .decoder import TransformerTransducerDecoder

class TransformerTransducer(nn.Module):
    """Implements the Transformer-Transducer model as described in
    https://arxiv.org/abs/2002.02562

    Args:

        in_features (int): The input feature size.

        n_classes (int): The number of classes/vocabulary.

        n_layers (int): The number of transformer encoder layers with truncated
        self attention.

        n_dec_layers (int): The number of layers in the decoder (predictor).

        d_model (int): The model dimensionality.

        ff_size (int): The feed forward inner layer dimensionality.

        h (int): The number of heads in the attention mechanism.

        joint_size (int): The joint layer feature size.

        enc_left_size (int): The size of the left window that each time step is
        allowed to look at in the encoder.

        enc_right_size (int): The size of the right window that each time step is
        allowed to look at in the encoder.

        dec_left_size (int): The size of the left window that each time step is
        allowed to look at in the decoder.

        dec_right_size (int): The size of the right window that each time step is
        allowed to look at in the decoder.

        p_dropout (float): The dropout rate.

        stride (int): The stride of the convolution layer in the prenet. Default 1.

        kernel_size (int): The kernel size of the convolution layer in the prenet. Default 1.

        masking_value (float, optional): The value to use for masking padded
        elements. Defaults to -1e15.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        n_layers: int,
        n_dec_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        joint_size: int,
        enc_left_size: int,
        enc_right_size: int,
        dec_left_size: int,
        dec_right_size: int,
        p_dropout: float,
        stride: int = 1,
        kernel_size: int = 1,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.encoder = TransformerTransducerEncoder(
            in_features=in_features,
            n_layers=n_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            left_size=enc_left_size,
            right_size=enc_right_size,
            p_dropout=p_dropout,
            stride=stride,
            kernel_size=kernel_size,
            masking_value=masking_value,
        )
        self.decoder = TransformerTransducerDecoder(
            vocab_size=n_classes,
            n_layers=n_dec_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            left_size=dec_left_size,
            right_size=dec_right_size,
            p_dropout=p_dropout,
            masking_value=masking_value,
        )
        self.audio_fc = nn.Linear(in_features=d_model, out_features=joint_size)
        self.text_fc = nn.Linear(in_features=d_model, out_features=joint_size)
        self.tanh = nn.Tanh()
        self.join_net = nn.Linear(in_features=joint_size, out_features=n_classes)

    def _join(self, encoder_out: Tensor, deocder_out: Tensor, training=True) -> Tensor:
        if training:
            encoder_out = encoder_out.unsqueeze(-2)
            deocder_out = deocder_out.unsqueeze(1)
        result = encoder_out + deocder_out
        result = self.tanh(result)
        result = self.join_net(result)
        return result
    
    def forward(
        self,
        speech: Tensor,
        speech_mask: Tensor,
        targets: Tensor,  # Đổi tên text -> targets (không chứa SOS)
        text_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass với blank token đầu decoder input

        Args:
            speech: [B, M, d]
            speech_mask: [B, M]
            targets: [B, N-1] (target không chứa SOS)
            text_mask: [B, N]
        
        Returns:
            logits: [B, M, N, C]
        """
        # Thêm blank token (0) đầu chuỗi
        blank = torch.zeros((targets.size(0), 1), device=targets.device).long()
        decoder_input = torch.cat([blank, targets], dim=1)  # [B, N]
        text_mask = torch.cat([torch.ones((targets.size(0), 1), device=targets.device).bool(), text_mask], dim=1)
        # Encoder
        speech_enc, speech_len = self.encoder(speech, speech_mask)
        
        # Decoder (dùng decoder_input đã thêm blank)
        text_dec, text_len = self.decoder(decoder_input, text_mask)
        
        # Project features
        speech_proj = self.audio_fc(speech_enc)
        text_proj = self.text_fc(text_dec)
        
        # Joint network
        logits = self._join(speech_proj, text_proj)
        
        return logits, speech_len, text_len