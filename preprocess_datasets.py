import argparse
import utils.config as cfg
from utils import objfile
import mindspore
from models import SMPL
import numpy as np
from utils.renderer import UVRenderer

from datasets.preprocess import \
    up_3d_extract, \
    lsp_dataset_original_extract, \
    process_dataset

import mindspore.context as context


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', default=True, action='store_true', help='Extract files needed for training')
    parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--gt_iuv', default=False, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--uv_type', type=str, default='BF', choices=['BF', 'SMPL'])

    args = parser.parse_args()

    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    # print("args = ", args)
    print("cfg.UP_3D_ROOT= ", cfg.UP_3D_ROOT)
    print("out_path = ", out_path)
    if args.train_files:

        # UP-3D dataset preprocessing (trainval set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'trainval')

        # LSP dataset original preprocessing (training set)
        lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, out_path)

    if args.eval_files:

        # h36m_extract(cfg.H36M_ROOT_ORIGIN, out_path, protocol=1, extract_img=True)
        # h36m_extract(cfg.H36M_ROOT_ORIGIN, out_path, protocol=2, extract_img=False)

        # LSP dataset preprocessing (test set)
        # lsp_dataset_extract(cfg.LSP_ROOT, out_path)

        # UP-3D dataset preprocessing (lsp_test set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')

    if args.gt_iuv:
        smpl = SMPL(model_file=cfg.SMPL_FILE)
        uv_type = args.uv_type
        print("uv_type = ", uv_type)
        uv_type = args.uv_type

        if uv_type == 'SMPL':
            data = objfile.read_obj_full('/data/users/user1/Master_04/Deco_MR/data/uv_sampler/smpl_fbx_template.obj')
        elif uv_type == 'BF':
            data = objfile.read_obj_full('/data/users/user1/Master_04/Deco_MR/data/uv_sampler/smpl_boundry_free_template.obj')

        vt = np.array(data['texcoords'])
        face = [f[0] for f in data['faces']]
        face = np.array(face) - 1
        vt_face = [f[2] for f in data['faces']]
        vt_face = np.array(vt_face) - 1
        renderer = UVRenderer(faces=face, tex=np.zeros([256, 256, 3]), vt=1 - vt, ft=vt_face)

        for dataset_name in ['up-3d', 'h36m-train']:
            process_dataset(dataset_name, is_train=True, uv_type=uv_type, smpl=smpl, renderer=renderer)


