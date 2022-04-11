"""
Copied from https://github.com/nkolot/GraphCMR
"""

from __future__ import division
import mindspore.ops as ops
from mindspore import Tensor as Tensor
from mindspore.dataset import GeneratorDataset
import mindspore
from random import shuffle

import mindspore.dataset
import torch


class RandomSampler(mindspore.dataset.RandomSampler):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        # print("data_source = ", len(data_source))
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            x = [i for i in range(len(self.data_source))]
            shuffle(x)
            self.dataset_perm = x
            y = [i for i in range(len(self.data_source))]
            shuffle(y)
            self.perm = y

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

class SequentialSampler(mindspore.dataset.SequentialSampler ):

    def __init__(self, data_source, checkpoint):
        self.data_source = data_source
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

class CheckpointDataLoader(GeneratorDataset):
    """
    Extends torch.utils.data.DataLoader to handle resuming training from an arbitrary point within an epoch.
    """
    def __init__(self, dataset, checkpoint=None, batch_size=1,
                 shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
                 timeout=0, worker_init_fn=None):
        #super(CheckpointDataLoader, self).__init__(dataset, sampler=sampler, shuffle=False, num_parallel_workers=num_workers)

        if shuffle:
            sampler = RandomSampler(dataset, checkpoint)
        else:
            sampler = SequentialSampler(dataset, checkpoint)

        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0
        super(CheckpointDataLoader, self).__init__(dataset, sampler=sampler, shuffle=False, num_parallel_workers=num_workers)

        #self.batch(batch_size)

# class CheckpointDataLoader(GeneratorDataset):
#     """
#     Extends torch.utils.data.DataLoader to handle resuming training from an arbitrary point within an epoch.
#     """
#     def __init__(self, dataset, checkpoint=None, batch_size=1,
#                  shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
#                  timeout=0, worker_init_fn=None):
#
#         if shuffle:
#             sampler = RandomSampler(dataset, checkpoint)
#         else:
#             sampler = SequentialSampler(dataset, checkpoint)
#
#         if checkpoint is not None:
#             self.checkpoint_batch_idx = checkpoint['batch_idx']
#         else:
#             self.checkpoint_batch_idx = 0
#
#         super(CheckpointDataLoader, self).__init__(dataset, sampler=sampler, shuffle=False, batch_size=batch_size, num_workers=num_workers,
#                                                    drop_last=drop_last, pin_memory=pin_memory, timeout=timeout, worker_init_fn=None)
