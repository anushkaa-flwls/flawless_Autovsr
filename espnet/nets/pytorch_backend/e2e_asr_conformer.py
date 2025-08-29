#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from .frontend.resnet import video_resnet
from .frontend.resnet1d import audio_resnet
from .ctc import CTC
from .encoder.conformer_encoder import ConformerEncoder
from .decoder.transformer_decoder import TransformerDecoder
from .nets_utils import make_non_pad_mask, th_accuracy
from .transformer.add_sos_eos import add_sos_eos
from .transformer.label_smoothing_loss import LabelSmoothingLoss
from .transformer.mask import target_mask
from ..scorers.ctc import CTCPrefixScorer


class E2E(torch.nn.Module):
    def __init__(self, odim, modality, ctc_weight=0.1, ignore_id=-1):
        super().__init__()

        self.modality = modality
        if modality == "audio":
            self.frontend = audio_resnet()
        elif modality == "video":
            self.frontend = video_resnet()

        self.proj_encoder = torch.nn.Linear(512, 768)

        self.encoder = ConformerEncoder(
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=12,
            cnn_module_kernel=31,
        )

        self.decoder = TransformerDecoder(
            odim=odim,
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=6,
        )

        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # loss
        self.ctc_weight = ctc_weight
        self.ctc = CTC(odim, 768, 0.1, reduce=True)
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)
  
    def scorers(self):
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths ,  audio_embeddings=None):  #torch.Size([4, 64000, 1]) Audio shape
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")

        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
    
        x = self.frontend(x)
        x = self.proj_encoder(x)
        x, _ = self.encoder(x, padding_mask)
        if audio_embeddings is  None:
            return x 
        
            
        # ctc loss
        loss_ctc, ys_hat = self.ctc(x, lengths, label)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, acc
