import models
import torch
import os

import numpy as np
from torch.utils.data.sampler import Sampler
import itertools
import random


def generate_model(option, ema=False):
    if option.model == 'PolypUU':
        model = getattr(models, option.model)(option.num_class, option.feat_level, option.dropout)
    else:
        model = getattr(models, option.model)(option.num_class)
    if option.use_gpu:
        model.cuda()

    if option.load_ckpt is not None:
        model_dict = model.state_dict()

        if option.select_checkpoint is not None:
            root_dir = option.select_checkpoint
        elif option.pretrain is not None:
            root_dir = option.pretrain
        else:
            root_dir = f"{option.dataset}_{option.suffix}"

        load_ckpt_path = os.path.join(
            option.checkpoints,
            root_dir,
            f'{option.method}_{option.model}',
            f'exp{option.expID}',
            f'{option.load_ckpt}.pth'
        )

        if os.path.isfile(load_ckpt_path):
            print(f'Loading {option.method} checkpoint: {load_ckpt_path}')
            checkpoint = torch.load(load_ckpt_path)
            new_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
            print('Done')
        else:
            print('No checkpoint found.')

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, total_count, primary_count, primary_batch_size, secondary_batch_size, shuffle=False):
        super().__init__(data_source=None)

        self.indices = list(range(total_count))
        if shuffle:
            random.shuffle(self.indices)

        self.primary_indices = self.indices[:primary_count]
        self.secondary_indices = self.indices[primary_count:]

        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# base_seed should be large enough to keep 0 and 1 bits balanced
def set_seed(inc, base_seed=2023):
    # cuDNN
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enable = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    os.environ['PYTHONHASHSEED'] = str(seed + 4)

    # # cuDNN
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    #
    # seed = base_seed + inc
    # random.seed(seed)
    # np.random.seed(seed + 1)
    # torch.manual_seed(seed + 2)
    # torch.cuda.manual_seed(seed + 3)
