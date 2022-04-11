"""
This file contains the definition of different heterogeneous datasets used for training
Codes are adapted from https://github.com/nkolot/GraphCMR
"""


from .base_dataset import BaseDataset











def create_dataset(dataset, options, use_IUV=False):
    # if dataset == 'all':
    #     return FullDataset(options, use_IUV=use_IUV)
    # elif dataset == 'itw':
    #     return ITWDataset(options, use_IUV=use_IUV)
    # elif dataset == 'h36m':
    #     return BaseDataset(options, 'h36m-train', use_IUV=use_IUV)
    # elif dataset == 'up-3d':
    #     return BaseDataset(options, 'up-3d', use_IUV=use_IUV)
    # elif dataset == 'mesh':
    #     return MeshDataset(options, use_IUV=use_IUV)
    # elif dataset == 'spin':
    #     return SPINDataset(options, use_IUV=use_IUV)
    # elif dataset == 'surreal':
    #     return SurrealDataset(options, use_IUV=use_IUV)
    # else:
    #     raise ValueError('Undefined dataset')
    if dataset == 'up-3d':
        return BaseDataset(options, 'up-3d', use_IUV=use_IUV)