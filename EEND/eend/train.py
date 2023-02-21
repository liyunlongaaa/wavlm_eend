#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini)
# Licensed under the MIT license.


from backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    pad_labels,
    pad_sequence,
    save_checkpoint,
)
from backend.updater import setup_optimizer, get_rate
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from torch.utils.data import DataLoader
from torch.distributed import is_initialized
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import random
import torch
import logging
import yamlargparse
import tqdm
import warnings
warnings.filterwarnings('ignore')

from s3prl import hub

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _ in batch], #feartures
            'ts': [t for _, t, _ in batch], #label
            'names': [r for _, _, r in batch]} # reco


def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    n_speakers: List[int],
    acum_metrics: Dict[str, float],
    vad_loss_weight: float,
    detach_attractor_loss: bool
) -> Tuple[torch.Tensor, Dict[str, float]]:
    y_pred, attractor_loss = model(input, labels, n_speakers, args)

    max_n_speakers = max(n_speakers)
    labels = pad_labels(labels, max_n_speakers)
    print(labels[0].shape, labels[1].shape,)
    labels = torch.stack(labels).to(args.device)
    print(y_pred, labels.shape)
    loss, standard_loss = model.get_loss(
        y_pred, labels, n_speakers, attractor_loss, vad_loss_weight,
        detach_attractor_loss)
    
    metrics = calculate_metrics(
        labels.detach(), y_pred.detach(), threshold=0.5)
    acum_metrics = update_metrics(acum_metrics, metrics)
    acum_metrics['loss'] += loss.item()
    acum_metrics['loss_standard'] += standard_loss.item()           #就只是pit loss
    acum_metrics['loss_attractor'] += attractor_loss.item()
    return loss, acum_metrics


def get_training_dataloaders(
    args: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    train_set = KaldiDiarizationDataset(
        args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,     #TODO : 了解怎么每帧加入上下文信息的
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,  #最后一个chunks保留的最短长度
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batchsize,
        collate_fn=_convert,
        num_workers=args.num_workers,
        shuffle=True,
        worker_init_fn=_init_fn,
    )

    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.dev_batchsize,
        collate_fn=_convert,
        num_workers=1,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    # Y_train, _, _ = train_set.__getitem__(0)
    # Y_dev, _, _ = dev_set.__getitem__(0)

    # assert Y_train.shape[1] == Y_dev.shape[1], \
    #     f"Train features dimensionality ({Y_train.shape[1]}) and \
    #     dev features dimensionality ({Y_dev.shape[1]}) differ."
    # assert Y_train.shape[1] == (
    #     args.feature_dim * (1 + 2 * args.context_size)), \
    #     f"Expected feature dimensionality of {args.feature_dim} \
    #     but {Y_train.shape[1]} found."

    return train_loader, dev_loader


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--num-frames', default=500, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--valid-data-dir',
                        help='kaldi-style data dir used for validation.')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument(
        '--attractor-loss-ratio', default=1.0, type=float,
        help='weighting parameter')
    attractor_args.add_argument(
        '--attractor-encoder-dropout', type=float)
    attractor_args.add_argument(
        '--attractor-decoder-dropout', type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', type=bool,
        help='If True, avoid backpropagation on attractor loss')
    
    #upstream upstream_trainable upstream
    parser.add_argument('--upstream_trainable', default=True, type=bool)
    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams with \"_local\" or \"_url\" postfix need local ckpt (-k) or config file (-g). '
        'Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. '
        'Please check upstream/README.md for details. '
        f"Available options in S3PRL: {upstreams}. "
    )
    parser.add_argument('-k', '--upstream_ckpt', metavar='{PATH,URL,GOOGLE_DRIVE_ID}', help='Only set when the specified upstream need it')
    parser.add_argument('-r', '--upstream_refresh', action='store_true', help='Re-download cached ckpts for on-the-fly upstream variants')
    parser.add_argument('--upstream_model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')
    parser.add_argument('-s', '--upstream_feature_selection', default='hidden_states', help='Specify the layer to be extracted as the representation')
    parser.add_argument('-l', '--upstream_layer_selection', type=int, help='Select a specific layer for the features selected by -s')
    parser.add_argument('--upstream_feature_normalize', action='store_true', help='Specify whether to normalize hidden features before weighted sum')
    parser.add_argument('--upstream_model_name', default="model.pt", help='The name of the model file in the HuggingFace Hub repo.')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.getLogger().setLevel(logging.INFO)
    logging.info(args)

    writer = SummaryWriter(f"{args.output_path}/tensorboard")

    train_loader, dev_loader = get_training_dataloaders(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    model = get_model(args)
    trainable_models = []
    trainable_paras = []
    
    if model.upstream.trainable:
            model.upstream.model.train()
            trainable_models.append(model.upstream.model)
            trainable_paras += list(model.upstream.model.parameters())
    else:
        model.upstream.model.eval()

    if model.featurizer.trainable:
            model.upstream.model.train()
            trainable_models.append(model.featurizer.model)
            trainable_paras += list(model.featurizer.model.parameters())
    else:
        model.featurizer.model.eval()
    
    trainable_models.append(model)
    trainable_paras += list(model.parameters())
    #print(len(trainable_models))
    if args.init_model_path == '':
        optimizer = setup_optimizer(args, model, trainable_models)
    else:
        model = average_checkpoints(
            args.device, model, args.init_model_path, args.init_epochs)
        optimizer = setup_optimizer(args, model, trainable_models)

    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)
    logging.info(f"#batches quantity for train: {train_batches_qty}")
    logging.info(f"#batches quantity for dev: {dev_batches_qty}")

    acum_train_metrics = new_metrics()
    acum_dev_metrics = new_metrics()

    if os.path.isfile(os.path.join(
            args.output_path, 'models', 'checkpoint_0.tar')):
        # Load latest model and continue from there
        directory = os.path.join(args.output_path, 'models')
        checkpoints = os.listdir(directory)
        paths = [os.path.join(directory, basename) for
                 basename in checkpoints if basename.startswith("checkpoint_")]
        latest = max(paths, key=os.path.getctime)
        epoch, model, optimizer, _ = load_checkpoint(args, latest, trainable_models)
        init_epoch = epoch
    else:
        init_epoch = 0
        # Save initial model
        save_checkpoint(args, init_epoch, model, optimizer, 0)

    for epoch in tqdm.tqdm(range(init_epoch, args.max_epochs)):
        model.upstream.model.train()
        model.featurizer.model.train()
        model.train()
        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            wavs = batch['xs']
            labels = batch['ts']

            #print(len(wavs), len(labels), wavs[0].shape)
            n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                     if t.sum() > 0 else 0 for t in labels])
            wavs = [torch.FloatTensor(wav).to(args.device) for wav in wavs]
            labels = [torch.FloatTensor(label).to(args.device) for label in labels]

            try:
                if model.upstream.trainable:
                    features = model.upstream.model(wavs)
                else:
                    with torch.no_grad():
                        features = model.upstream.model(wavs)
                features = model.featurizer.model(wavs, features)  #TODO 出來有的會少一行是啥情況

                #padding到(B, num_frames, dim)
                features, labels = pad_sequence(features, labels, args.num_frames, args.device)  #args.num_frames定義了T的最大長度
                features = torch.stack(features).to(args.device)

                #padding到(B, num_frames, max_n_speakers)
                max_n_speakers = max(n_speakers)
                labels = pad_labels(labels, max_n_speakers)
                labels = torch.stack(labels).to(args.device)
                
                y_pred, attractor_loss = model(features, labels, n_speakers, args)
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f'[Runner] - CUDA out of memory at step {i}')
                    if is_initialized():
                        raise
                    with torch.cuda.device(args.device):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise
            loss, standard_loss = model.get_loss(
                y_pred, labels, n_speakers, attractor_loss, args.vad_loss_weight,
                args.detach_attractor_loss)
            metrics = calculate_metrics(
            labels.detach(), y_pred.detach(), threshold=0.5)
            acum_train_metrics = update_metrics(acum_train_metrics, metrics)
            acum_train_metrics['loss'] += loss.item()
            acum_train_metrics['loss_standard'] += standard_loss.item()           #就只是pit loss
            acum_train_metrics['loss_attractor'] += attractor_loss.item()

            if i % args.log_report_batches_num == \
                    (args.log_report_batches_num-1):
                for k in acum_train_metrics.keys():
                    writer.add_scalar(
                        f"train_{k}",
                        acum_train_metrics[k] / args.log_report_batches_num,
                        epoch * train_batches_qty + i)
                writer.add_scalar(
                    "lrate",
                    get_rate(optimizer),
                    epoch * train_batches_qty + i)
                acum_train_metrics = reset_metrics(acum_train_metrics)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_paras, args.gradclip)
            optimizer.step()
        save_checkpoint(args, epoch+1, model, optimizer, loss)

        with torch.no_grad():
            model.upstream.model.eval()
            model.featurizer.model.eval()
            model.eval()
            for i, batch in enumerate(tqdm.tqdm(dev_loader)):
                
                try:
                    wavs = batch['xs']
                    labels = batch['ts']
                    #print(len(wavs), len(labels))
                    n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                            if t.sum() > 0 else 0 for t in labels])
                    wavs = [torch.FloatTensor(wav).to(args.device) for wav in wavs]
                    labels = [torch.FloatTensor(label).to(args.device) for label in labels]
                    if model.upstream.trainable:
                        features = model.upstream.model(wavs)
                    else:
                        with torch.no_grad():
                            features = model.upstream.model(wavs)
                    
                    features = model.featurizer.model(wavs, features)
                    #padding到(B, num_frames, dim)
                    features, labels = pad_sequence(features, labels, args.num_frames, args.device)  #args.num_frames定義了T的最大長度
                    features = torch.stack(features).to(args.device)

                    #padding到(B, num_frames, max_n_speakers)
                    max_n_speakers = max(n_speakers)
                    labels = pad_labels(labels, max_n_speakers)
                    labels = torch.stack(labels).to(args.device)
                    
                    y_pred, attractor_loss = model(features, labels, n_speakers, args)

                except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f'[Runner] - CUDA out of memory at step {i}')
                            if is_initialized():
                                raise
                            with torch.cuda.device(args.device):
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                loss, standard_loss = model.get_loss(
                    y_pred, labels, n_speakers, attractor_loss, args.vad_loss_weight,
                    args.detach_attractor_loss)
                metrics = calculate_metrics(
                labels.detach(), y_pred.detach(), threshold=0.5)
                acum_dev_metrics = update_metrics(acum_dev_metrics, metrics)
                acum_dev_metrics['loss'] += loss.item()
                acum_dev_metrics['loss_standard'] += standard_loss.item()           #就只是pit loss
                acum_dev_metrics['loss_attractor'] += attractor_loss.item()

        for k in acum_dev_metrics.keys():
            writer.add_scalar(
                f"dev_{k}", acum_dev_metrics[k] / dev_batches_qty,
                epoch * dev_batches_qty + i)
        acum_dev_metrics = reset_metrics(acum_dev_metrics)
