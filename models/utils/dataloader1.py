"""
Copied from https://github.com/nkolot/GraphCMR
"""

from __future__ import division
import mindspore.ops as ops
from mindspore import Tensor as Tensor
from mindspore.dataset import GeneratorDataset,NumpySlicesDataset,BatchDataset
import mindspore
from random import shuffle
import mindspore.dataset as ds


class RandomSampler(ds.DistributedSampler):

    def __init__(self, data_source, checkpoint):
        super(RandomSampler, self).__init__(num_shards=10,shard_id=0,shuffle=False,num_samples=10)
        self.data_source = data_source
        #print("data_source = ", len(data_source))
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size'] * checkpoint['batch_idx']:]
        else:
            x = [i for i in range(len(self.data_source))]
            shuffle(x)
            self.dataset_perm = x
            y = [i for i in range(len(self.data_source))]
            shuffle(y)
            self.perm = y
        #print(self.perm)

    def __iter__(self):
        for i in self.perm:
            yield i
    def __len__(self):
        return len(self.perm)


class SequentialSampler(ds.DistributedSampler):

    def __init__(self, data_source, checkpoint):
        super(SequentialSampler, self).__init__(len(data_source)/2,0,True,2)
        self.data_source = data_source
        #print(len(self.data_source))
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            self.perm = self.dataset_perm[checkpoint['batch_size'] * checkpoint['batch_idx']:]
        else:
            self.dataset_perm = list(range(len(self.data_source)))
            self.perm = self.dataset_perm

    def __iter__(self):
        # return iter(self.perm)
        for i in self.perm:
            yield i

    def __len__(self):
        return len(self.perm)


class CheckpointDataLoader(GeneratorDataset):
    """
    Extends torch.utils.data.DataLoader to handle resuming training from an arbitrary point within an epoch.
    """

    def __init__(self, dataset, checkpoint=None, batch_size=1,
                 shuffle=False, num_workers=0, pin_memory=False, drop_last=True,
                 timeout=0, worker_init_fn=None):
        # super(CheckpointDataLoader, self).__init__(dataset, sampler=sampler, shuffle=False, num_parallel_workers=num_workers)

        if shuffle:
            sampler = RandomSampler(dataset, checkpoint)
        else:
            sampler = SequentialSampler(dataset, checkpoint)
        #print("sampler长度:")
        #print(len(sampler))
        #print("hhhh")
        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0
        #print("dataset长度",len(dataset))
        super(CheckpointDataLoader, self).__init__(source=dataset, column_names=[
            "scale",
            "center",
            "orig_shape",
            "img_orig",
            "img",
            "imgname",
            "has_smpl",
            "pose",
            "betas",
            "has_pose_3d",
            "pose_3d",
            "keypoints",
            "keypoints_smpl",
            "pose_3d_smpl",
            "has_pose_3d_smpl",
            "maskname",
            "partname",
            "gender",
            "gt_iuv",
            "iuvname",
            "has_dp",
            "fit_joint_error",
        ], num_parallel_workers=1,)


'''
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

        self.batch(batch_size)
        
        
        
        
        
        
        
        
        
        
        
        
                    "scale",
            "center",
            "orig_shape",
            #"img_orig",
            #"img",
            "imgname",
            "has_smpl",
            "pose",
            "betas",
            "has_pose_3d",
            "pose_3d",
            "keypoints",
            "keypoints_smpl",
            "pose_3d_smpl",
            "has_pose_3d_smpl",
            "maskname",
            "partname",
            "gender",
            "gt_iuv",
            "iuvname",
            "has_dp",
            "fit_joint_error",
'''