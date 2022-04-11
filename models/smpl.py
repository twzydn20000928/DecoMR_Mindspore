from __future__ import division

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context
import mindspore.nn as nn
import mindspore.numpy

import numpy as np
from .geometric_layers import rodrigues

try:
    import cPickle as pickle
except ImportError:
    import pickle

import utils.config as cfg

class SMPL(nn.Cell):
# class SMPL():
    def __init__(self, model_file=cfg.SMPL_FILE):
        super(SMPL, self).__init__()
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')
        # print("smpl_model = ", smpl_model)
        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = Tensor([row, col])
        v = Tensor(data,mindspore.float32)
        J_regressor_shape = [24, 6890]
        # print("i = ",i,"\nv = ", v, "\nJ_regressor_shape = ", J_regressor_shape)
        #设置持久缓冲区，不是模型的参数，缓冲区可以使用给定的名称作为属性访问。
        
        print(context.get_context("device_target"))

        sparse_to_dense = ops.SparseToDense()
        # self.register_buffer('J_regressor', sparse_to_dense(i.T, v, J_regressor_shape))
        self.J_regressor = sparse_to_dense(i.T, v, J_regressor_shape)
        print(self.J_regressor.shape)
        
        print(context.get_context("device_target"))

        # self.register_buffer('weights', Tensor(smpl_model['weights']),mindspore.float32)
        # self.register_buffer('posedirs', Tensor(smpl_model['posedirs']),mindspore.float32)
        # self.register_buffer('v_template', Tensor(smpl_model['v_template']),mindspore.float32)
        # self.register_buffer('shapedirs', Tensor(np.array(smpl_model['shapedirs'])),mindspore.float32)
        # self.register_buffer('faces', Tensor.from_numpy(smpl_model['f'].astype("float32")))
        # self.register_buffer('kintree_table', Tensor.from_numpy(smpl_model['kintree_table'].astype("int64")))
        # self.register_buffer('parent', Tensor(
        #     [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))
        self.kintree_table = Tensor.from_numpy(smpl_model['kintree_table'].astype("int64"))
        id_to_col = {self.kintree_table[1, i].asnumpy().item(): i for i in range(self.kintree_table.shape[1])}
        self.weights = Tensor(smpl_model['weights'],mindspore.float32)
        self.shapedirs = Tensor((np.array(smpl_model['shapedirs'])),mindspore.float32)
        self.v_template = Tensor((np.array(smpl_model['v_template'])), mindspore.float32)
        self.posedirs = Tensor((np.array(smpl_model['posedirs'])), mindspore.float32)
        self.parent = Tensor([id_to_col[self.kintree_table[0, it].asnumpy().item()] for it in range(1, self.kintree_table.shape[1])])
        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = mindspore.numpy.zeros(self.pose_shape)
        self.beta = mindspore.numpy.zeros(self.beta_shape)
        self.translation = mindspore.numpy.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        J_regressor_extra = Tensor.from_numpy(np.load(cfg.JOINT_REGRESSOR_TRAIN_EXTRA)).astype("float32")
        # self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.J_regressor_extra = J_regressor_extra
        self.joints_idx = cfg.JOINTS_IDX

        # This is the h36m regressor used in Graph-CMR(https://github.com/nkolot/GraphCMR)
        # self.register_buffer('h36m_regressor_cmr', Tensor(np.load(cfg.JOINT_REGRESSOR_H36M)),mindspore.float32)
        # self.register_buffer('lsp_regressor_cmr', Tensor(np.load(cfg.JOINT_REGRESSOR_H36M),mindspore.float32)[cfg.H36M_TO_J14])

        # This is another lsp joints regressor, we use it for training and evaluation
        # self.register_buffer('lsp_regressor_eval', Tensor(np.load(cfg.LSP_REGRESSOR_EVAL),mindspore.float32).permute(1, 0))

        self.lsp_regressor_eval = Tensor(np.load(cfg.LSP_REGRESSOR_EVAL),mindspore.float32).transpose([1, 0])
        # We hope the training and evaluation regressor for the lsp joints to be consistent,
        # so we replace parts of the training regressor used in Graph-CMR.
        op = ops.Concat(0)
        train_regressor = op([self.J_regressor, self.J_regressor_extra])
        train_regressor = train_regressor[[cfg.JOINTS_IDX]].copy()
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        train_regressor[idx] = self.lsp_regressor_eval
        # self.register_buffer('train_regressor', train_regressor)


    def construct(self, pose, beta):
        # device = pose.device
        print("smpl")
        batch_size = pose.shape[0]

        v_template = self.v_template[None, :]
                # expand_dims = ops.ExpandDims()
                # posedirs = expand_dims(posedirs, 0)
                # posedirs = posedirs.repeat(batch_size, 0)
        '''下面这句话有问题 10.4'''
        shapedirs_shape = self.shapedirs.shape
        print("shapedirs_shape = ", shapedirs_shape)
        shapedirs = self.shapedirs.view(-1, 10)[None, :]
        # expand_dims = ops.ExpandDims()
        # shapedirs = expand_dims(shapedirs, 0)
        shapedirs = shapedirs.repeat(batch_size, 0)
        # shapedirs = self.shapedirs.view(-1, 10)[None, :].repeat(batch_size, axis=0)
        # beta = beta[:, :, None]
        expand_dims = ops.ExpandDims()
        beta = expand_dims(beta, -1)
        mul = ops.matmul(shapedirs, beta).view(-1, 6890, 3)
        v_shaped = ops.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template

        # batched sparse matmul not supported in pytorch
        # print("smplhhh")
        J = []
        for i in range(batch_size):
            J.append(ops.matmul(self.J_regressor, v_shaped[i]))
        J = ops.Stack(axis=0)(J)

        # input it rotmat: (bs,24,3,3)
        # print("pose.ndim = ", pose.ndim)
        if pose.ndim == 4:
            R = pose
        # input it rotmandimensiont: (bs,72)
        elif pose.ndim == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)

            R = rodrigues(pose_cube)
            R = R.view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        # I_cube = torch.eye(3)[None, None, :].to(device)
        # print("smplhhh----hhh")
        eye = ops.Eye()
        x = eye(3, 3, mindspore.float32)
        expand_dims = ops.ExpandDims()
        x = expand_dims(x, 0)
        I_cube = expand_dims(x, 0)
        # I_cube = x[None, None, :].astype(pose.dtype)

        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)
        expand_dims = ops.ExpandDims()
        posedirs = expand_dims(posedirs, 0)
        posedirs = posedirs.repeat(batch_size, 0)
        x = ops.matmul(posedirs, lrotmin[:, :, None])
        x = x.view(-1, 6890, 3)
        expand_dims = ops.ExpandDims()
        lrotmin = expand_dims(lrotmin, -1)
        mul = ops.matmul(posedirs, lrotmin)
        mul_view = mul.view(-1, 6890, 3)
        # v_posed = v_shaped + ops.matmul(posedirs, lrotmin).view(-1, 6890, 3)
        v_posed = v_shaped + mul_view
        J_ = J.copy()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        expand_dims = ops.ExpandDims()
        J_ = expand_dims(J_, -1)
        G_ = ops.Concat(axis=-1)([R, J_])
        # pad_row = torch.FloatTensor([0,0,0,1]).to(device).view(1,1,1,4).expand(batch_size, 24, -1, -1)
        pad_row = mindspore.Tensor([0, 0, 0, 1], dtype=pose.dtype).view(1, 1, 1, 4).repeat(batch_size, axis=0).repeat(24,
                                                                                                                      axis=1)

        G_ = ops.Concat(axis=2)([G_, pad_row])
        G = G_.copy()

        for i in range(1, 24):
            G[:, i, :, :] = ops.matmul(G[:, self.parent[i - 1], :, :], G_[:, i, :, :])
        # rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = np.zeros((batch_size, 24, 1), dtype=np.float32)
        zeros = Tensor(zeros, dtype=pose.dtype)
        rest = ops.Concat(axis=2)([J, zeros]).view(batch_size, 24, 4, 1)

        # zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        zeros = np.zeros((batch_size, 24, 4, 3), dtype=np.float32)
        zeros = Tensor(zeros, dtype=pose.dtype)
        rest = ops.Concat(axis=-1)([zeros, rest])
        rest = ops.matmul(G, rest)
        G = G - rest
        # x = G.transpose(1, 0, 2, 3).view(24, -1)
        # y = ops.matmul(self.weights, G.transpose(1, 0, 2, 3).view(24, -1))
        # T = ops.matmul(self.weights, G.transpose(1, 0, 2, 3).view(24, -1)).view(6890, batch_size, 4, 4)
        T = ops.matmul(self.weights, G.transpose(1, 0, 2, 3).view(24, -1)).view(6890, batch_size, 4, 4).transpose(1, 0, 2, 3)
        rest_shape_h = ops.Concat(axis=-1)([v_posed, ops.OnesLike()(v_posed)[:, :, [0]]])
        v = ops.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    # The function used in Graph-CMR, outputting the 24 training joints.
    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = mindspore.Tensor(np.einsum('bik,ji->bjk', vertices, self.J_regressor))
        joints_extra = mindspore.Tensor(np.einsum('bik,ji->bjk', vertices, self.J_regressor_extra))
        joints = ops.Concat(axis=1)((joints, joints_extra))
        joints = joints[:, cfg.JOINTS_IDX]
        return joints

    # The function used in Graph-CMR, get 38 joints.
    def get_full_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = mindspore.Tensor(np.einsum('bik,ji->bjk', vertices, self.J_regressor))
        joints_extra = mindspore.Tensor(np.einsum('bik,ji->bjk', vertices, self.J_regressor_extra))
        joints = ops.Concat(axis=1)((joints, joints_extra))
        return joints

    # Get 14 lsp joints use the joint regressor provided by CMR.
    def get_lsp_joints(self, vertices):
        joints = ops.matmul(self.lsp_regressor_cmr[None, :], vertices)
        return joints

    # Get the joints defined by SMPL model.
    def get_smpl_joints(self, vertices):
        """
        This method is used to get the SMPL model joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = mindspore.Tensor(np.einsum('bik,ji->bjk', vertices, self.J_regressor))
        return joints  # .contiguous()

    # Get 24 training joints using the evaluation LSP joint regressor.
    def get_train_joints(self, vertices):
        """
        This method is used to get the training 24 joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = ops.matmul(self.train_regressor[None, :], vertices)
        return joints

    # Get 14 lsp joints for the evaluation.
    def get_eval_joints(self, vertices):
        """
        This method is used to get the 14 eval joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 14, 3)
        """
        joints = ops.matmul(self.lsp_regressor_eval[None, :], vertices)
        return joints









