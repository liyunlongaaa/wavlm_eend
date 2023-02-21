#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from os.path import isfile, join
import sys

from backend.losses import (
    pit_loss_multispk,
    vad_loss,
)
from backend.updater import (
    NoamOpt,
    setup_optimizer,
)
from pathlib import Path
from torch.nn import Module, ModuleList
from types import SimpleNamespace
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import s3prl
from s3prl.upstream.wav2vec2 import wav2vec2_model
from s3prl.upstream.wavlm import WavLM
from s3prl import hub

from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.upstream.interfaces import Featurizer
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict

Upstream = getattr(hub, 'wavlm')

"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""

class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces
        

class EncoderDecoderAttractor(Module):
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        encoder_dropout: float,
        decoder_dropout: float,
        detach_attractor_loss: bool,
    ) -> None:
        super(EncoderDecoderAttractor, self).__init__()
        self.device = device
        self.encoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            batch_first=True,                     #
            device=self.device)
        self.decoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=decoder_dropout,
            batch_first=True,
            device=self.device)
        self.counter = torch.nn.Linear(n_units, 1, device=self.device)
        self.n_units = n_units
        self.detach_attractor_loss = detach_attractor_loss

    def forward(self, xs: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:   #同时重写forward和call，想调用forward只能显试调用
        _, (hx, cx) = self.encoder.to(self.device)(xs.to(self.device))

        attractors, (_, _) = self.decoder.to(self.device)(
            zeros.to(self.device),                             #输入是0，hx输出（即下一时刻输入）永远是0，单cx不是
            (hx.to(self.device), cx.to(self.device))
        )
        return attractors    #forward的attractors shape 是 (B, zeros.shape[1] , n_units)

    def estimate(
        self,
        xs: torch.Tensor,
        max_n_speakers: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors from embedding sequences
         without prior knowledge of the number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          max_n_speakers (int)
        Returns:
          attractors: (B,T, max_n_speakers)-shaped attractors
          probs: List of attractor existence probabilities  [[p1, p2, ..., p_max_n_speakers], ...] batch个，用这个函数来确定人数 （根据概率阈值）
        """
        zeros = torch.zeros((xs.shape[0], max_n_speakers, self.n_units))
        attractors = self.forward(xs, zeros)    
        probs = [torch.sigmoid(
            torch.flatten(self.counter.to(self.device)(att)))
            for att in attractors]

        return attractors, probs

    def __call__(
        self,
        xs: torch.Tensor,
        n_speakers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors and loss from embedding sequences
        with given number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          n_speakers: List of number of speakers, or None if the number
                                of speakers is unknown (ex. test phase)
        Returns:
          loss: Attractor existence loss
          attractors: List of (N,D)-shaped attractors
        """

        max_n_speakers = max(n_speakers)
        if self.device == torch.device("cpu"):
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units))          # +1是为了让模型学会在后面有位置的情况下，终止，即确定人数
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)                   #目的是想让EDA 学会估计人数. (B, max_n_speakers + 1)
                for n_spk in n_speakers]))
        else:
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units),
                device=torch.device("cuda"))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers])).to(torch.device("cuda"))

        attractors = self.forward(xs, zeros)

        if self.detach_attractor_loss:   #默认是False
            attractors = attractors.detach()
        logit = torch.cat([
            torch.reshape(self.counter(att), (-1, max_n_speakers + 1))
            for att, n_spk in zip(attractors, n_speakers)])  #(B, max_n_speakers + 1)
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        # The final attractor does not correspond to a speaker so remove it
        attractors = attractors[:, :-1, :]    #不要最后一列
        return loss, attractors    # (B, max_n_speakers, dim)


class MultiHeadSelfAttention(Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linearQ = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearK = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearV = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearO = torch.nn.Linear(n_units, n_units, device=self.device)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) \
            / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = F.dropout(self.att, self.dropout)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(Module):
    """ Positionwise feed-forward layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        d_units: int,
        dropout: float
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(n_units, d_units, device=self.device)
        self.linear2 = torch.nn.Linear(d_units, n_units, device=self.device)
        self.dropout = dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.dropout(F.relu(self.linear1(x)), self.dropout))


class TransformerEncoder(Module):
    def __init__(
        self,
        device: torch.device,
        idim: int,
        n_layers: int,
        n_units: int,
        e_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.linear_in = torch.nn.Linear(idim, n_units, device=self.device)
        self.lnorm_in = torch.nn.LayerNorm(n_units, device=self.device)
        self.n_layers = n_layers
        self.dropout = dropout
        for i in range(n_layers):
            setattr(
                self,
                '{}{:d}'.format("lnorm1_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("self_att_", i),
                MultiHeadSelfAttention(self.device, n_units, h, dropout)
            )
            setattr(
                self,
                '{}{:d}'.format("lnorm2_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("ff_", i),
                PositionwiseFeedForward(self.device, n_units, e_units, dropout)
            )
        self.lnorm_out = torch.nn.LayerNorm(n_units, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)


class TransformerEDADiarization(Module):

    def __init__(
        self,
        args
        # device: torch.device,
        # in_size: int,
        # n_units: int,
        # e_units: int,
        # n_heads: int,
        # n_layers: int,
        # dropout: float,
        # vad_loss_weight: float,
        # attractor_loss_ratio: float,
        # attractor_encoder_dropout: float,
        # attractor_decoder_dropout: float,
        # detach_attractor_loss: bool,
    ) -> None:
        """ Self-attention-based diarization model.
        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          vad_loss_weight (float) : weight for vad_loss
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        self.args = args
        self.device = args.device
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        super(TransformerEDADiarization, self).__init__()

        # self.upstream = Upstream(
        #     ckpt = None,
        #     model_config = None,
        #     refresh = None,
        # )
        # device=args.device,
            # in_size=args.feature_dim,
            # n_units=args.hidden_size,
            # e_units=args.encoder_units,
            # n_heads=args.transformer_encoder_n_heads,
            # n_layers=args.transformer_encoder_n_layers,
            # dropout=args.transformer_encoder_dropout,
            # attractor_loss_ratio=args.attractor_loss_ratio,
            # attractor_encoder_dropout=args.attractor_encoder_dropout,
            # attractor_decoder_dropout=args.attractor_decoder_dropout,
            # detach_attractor_loss=args.detach_attractor_loss,
            # vad_loss_weight=args.vad_loss_weight,
        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()

        self.enc = TransformerEncoder(
            self.device, self.featurizer.model.output_dim, args.transformer_encoder_n_layers, args.hidden_size, args.encoder_units, args.transformer_encoder_n_heads, args.transformer_encoder_dropout
        )  ## output: (BT, F)， 和nlp不同
        self.eda = EncoderDecoderAttractor(
            self.device,
            args.hidden_size,
            args.attractor_encoder_dropout,
            args.attractor_decoder_dropout,
            args.detach_attractor_loss,
        )
        self.attractor_loss_ratio = args.attractor_loss_ratio
        self.vad_loss_weight = args.vad_loss_weight

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)
    
    def _get_upstream(self):

        Upstream = getattr(hub, self.args.upstream)
        ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
            interfaces = ["get_downsample_rates"]
        )


    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )

    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: [(T, E), ...]
        emb = emb.reshape(pad_shape[0], pad_shape[1], -1)   
        return emb #（B, T, E）

    def estimate_sequential(   #只在infer时候用到
        self,
        xs: torch.Tensor,
        args: SimpleNamespace
    ) -> List[torch.Tensor]:
        assert args.estimate_spk_qty_thr != -1 or \
            args.estimate_spk_qty != -1, \
            "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
            arguments have to be defined."
        emb = self.get_embeddings(xs)
        ys_active = []
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.eda.estimate(
                torch.stack([e[order] for e, order in zip(emb, orders)]))  # (B, T, max_n_speakers) , [[P1, ...,p_max_n_speakers], ...] B个
        else:
            attractors, probs = self.eda.estimate(emb)
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))  #(B, T, max_n_speakers)       <= （B, T, dim）@ (B, dim, max_n_speakers) S就是max_n_speakers 
        ys = [torch.sigmoid(y) for y in ys]
        for p, y in zip(probs, ys):
            if args.estimate_spk_qty != -1:     #如果是说话人数给定
                sorted_p, order = torch.sort(p, descending=True)       #order 是排序之前元素所在的位置，从大到小排列
                ys_active.append(y[:, order[:args.estimate_spk_qty]])   #选概率最大的前args.estimate_spk_qty个
            elif args.estimate_spk_qty_thr != -1: #用阈值估计说话人数
                silence = np.where(
                    p.data.to("cpu") < args.estimate_spk_qty_thr)[0]
                n_spk = silence[0] if silence.size else None      #选第一个小于thr之前的所以attractors, 全大于的话就全选
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError(
                    'estimate_spk_qty or estimate_spk_qty_thr needed.')
        return ys_active   # [[:n_spk], ...]

    def forward(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
        n_speakers: List[int],
        args: SimpleNamespace
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # if self.upstream.trainable:
        #     features = self.upstream.model(xs)
        # else:
        #     with torch.no_grad():
        #         features = self.upstream.model(xs)

        # features = self.featurizer.model(xs, features)

        # features, ts = pad_sequence(features, ts, args.num_frames)  #args.num_frames定義了T的最大長度
        # features = torch.stack(features).to(args.device)
        
        emb = self.get_embeddings(xs)
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)   #打乱时序作为eda输入
            attractor_loss, attractors = self.eda(   
                torch.stack([e[order] for e, order in zip(emb, orders)]),
                n_speakers)
        else:
            attractor_loss, attractors = self.eda(emb, n_speakers)

        # ys: (B, T, S)       <= （B, T, dim）@ (B, dim, S) S就是max_n_speakers 
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))

        return ys, attractor_loss

    def get_loss(
        self,
        ys: torch.Tensor,   #y_pred  (B, T, S) 
        target: torch.Tensor,
        n_speakers: List[int],
        attractor_loss: torch.Tensor,
        vad_loss_weight: float,
        detach_attractor_loss: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_n_speakers = max(n_speakers)

        ts_padded = pad_labels(target, max_n_speakers) #变为列表 [f1, f2, ..]
        ts_padded = torch.stack(ts_padded)  #在说话人人数确定且都不变的情况下，ts_padded target的shape一样

        ys_padded = pad_labels(ys, max_n_speakers)
        ys_padded = torch.stack(ys_padded)

        loss = pit_loss_multispk(
            ys_padded, ts_padded, n_speakers, detach_attractor_loss)    # TODO 这个之后要搞懂
        vad_loss_value = vad_loss(ys, target)   # TODO 这个之后要搞懂，就是希望模型学会判断哪里有声音，哪里没有

        return loss + vad_loss_value * vad_loss_weight + \
            attractor_loss * self.attractor_loss_ratio, loss   


def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding  用-1填充到speaker-dim
            ts_padded.append(torch.cat((t, -1 * torch.ones((
                t.shape[0], out_size - t.shape[1]))), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded


def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    seq_len: int ,
    device
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        "{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        #print(features[i].shape[0], labels[i].shape[0])
        # assert features[i].shape[0] == labels[i].shape[0], (
        #     f"Length of features and labels were expected to match but got "
        #     "{features[i].shape[0]} and {labels[i].shape[0]}")
        if features[i].shape[0] != labels[i].shape[0]:
            #print('features[i].shape[0] != labels[i].shape[0]', features[i].shape[0], labels[i].shape[0])  # TODO
            features[i] = torch.cat((features[i], features[i][-1:].repeat((labels[i].shape[0] - features[i].shape[0], 1))))

        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i], -torch.ones((
                extend, features[i].shape[1])).to(device)), dim=0))
            labels_padded.append(torch.cat((labels[i], -torch.ones((
                extend, labels[i].shape[1])).to(device)), dim=0))
        elif length > seq_len:
            raise (f"Sequence of length {length} was received but only "
                   "{seq_len} was expected.")
        else:
            features_padded.append(features[i])
            labels_padded.append(labels[i])
    return features_padded, labels_padded


def save_checkpoint(
    args,
    epoch: int,
    model: Module,
    optimizer: NoamOpt,
    loss: torch.Tensor
) -> None:
    Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'Upstream': model.upstream.model.state_dict(),
        'Featurizer': model.featurizer.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        f"{args.output_path}/models/checkpoint_{epoch}.tar"
    )
    

def load_checkpoint(args: SimpleNamespace, filename: str, trainable_models):
    model = get_model(args)
    optimizer = setup_optimizer(args, model, trainable_models)

    assert isfile(filename), \
        f"File {filename} does not exist."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.upstream.model.load_state_dict(checkpoint['Upstream'])
    model.featurizer.model.load_state_dict(checkpoint['Featurizer'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, model, optimizer, loss


def load_initmodel(args: SimpleNamespace):
    return load_checkpoint(args, args.initmodel)


def get_model(args: SimpleNamespace) -> Module:
    if args.model_type == 'TransformerEDA':
        model = TransformerEDADiarization(
            args
            # device=args.device,
            # in_size=args.feature_dim,
            # n_units=args.hidden_size,
            # e_units=args.encoder_units,
            # n_heads=args.transformer_encoder_n_heads,
            # n_layers=args.transformer_encoder_n_layers,
            # dropout=args.transformer_encoder_dropout,
            # attractor_loss_ratio=args.attractor_loss_ratio,
            # attractor_encoder_dropout=args.attractor_encoder_dropout,
            # attractor_decoder_dropout=args.attractor_decoder_dropout,
            # detach_attractor_loss=args.detach_attractor_loss,
            # vad_loss_weight=args.vad_loss_weight,
        )
    else:
        raise ValueError('Possible model_type is "TransformerEDA"')
    return model


def average_checkpoints(
    device: torch.device,
    model: Module,
    models_path: str,
    epochs: str
) -> Module:
    epochs = parse_epochs(epochs)
    states_dict_list = []
    u_states_dict_list = []
    f_states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        copy_upstream = copy.deepcopy(model.upstream.model)
        copy_featurizer = copy.deepcopy(model.featurizer.model)
        checkpoint = torch.load(join(
            models_path,
            f"checkpoint_{e}.tar"), map_location=device)
        copy_model.load_state_dict(checkpoint['model_state_dict'])
        copy_upstream.load_state_dict(checkpoint['Upstream'])
        copy_featurizer.load_state_dict(checkpoint['Featurizer'])

        states_dict_list.append(copy_model.state_dict())
        u_states_dict_list.append(copy_upstream.state_dict())
        f_states_dict_list.append(copy_featurizer.state_dict())

    avg_state_dict = average_states(states_dict_list, device)
    avg_u_state_dict = average_states(u_states_dict_list, device)
    avg_f_state_dict = average_states(f_states_dict_list, device)
    #avg_model = copy.deepcopy(model)
    model.load_state_dict(avg_state_dict)
    model.upstream.model.load_state_dict(avg_u_state_dict)
    model.featurizer.model.load_state_dict(avg_f_state_dict)
    return model


def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


def parse_epochs(string: str) -> List[int]:
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(int(p))
    return res
