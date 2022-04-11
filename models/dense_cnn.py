import mindspore
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor
import numpy as np
from mindspore import ops
import sys
from mindspore import Parameter

from models.resnet import resnet50backbone
from models.layer import ConvBottleNeck, HgNet
from models.ResizeBilinear import *
import os
from models.uv_generator import Index_UV_Generator
from utils.objfile import read_obj


def warp_feature(dp_out, feature_map, uv_res):
    assert dp_out.shape[0] == feature_map.shape[0]
    assert dp_out.shape[2] == feature_map.shape[2]
    assert dp_out.shape[3] == feature_map.shape[3]

    expand_dims = ops.ExpandDims()
    dp_mask = expand_dims(dp_out[:, 0], 1)
    dp_uv = dp_out[:, 1:]
    thre = 0.0005
    B, C, H, W = feature_map.shape
    # device =feature_map.device

    index_batch = mnp.arange(0, B, dtype=mindspore.int64)[:, None, None].repeat(H, 1).repeat(W, 2)
    index_batch = index_batch.view(-1).astype("int32")  # ,device=device

    tmp_x = mnp.arange(0, W, dtype=mindspore.int32)  # device=device,
    tmp_y = mnp.arange(0, H, dtype=mindspore.int32)  # device=device,

    meshgrid = ops.Meshgrid(indexing="ij")
    y, x = meshgrid((tmp_y, tmp_x))

    y = ops.stop_gradient(y)
    x = ops.stop_gradient(x)

    # y,x = np.meshgrid(tmp_y.asnumpy(), tmp_x.asnumpy(),indexing="ij")
    # y=mindspore.Tensor(y)
    # x=mindspore.Tensor(x)
    y = y.view(-1).repeat([B])
    x = x.view(-1).repeat([B])

    conf = dp_mask[index_batch, 0, y, x]
    valid = conf > thre

    index_batch = mindspore.Tensor(index_batch.asnumpy()[valid.asnumpy()])
    x = mindspore.Tensor(x.asnumpy()[valid.asnumpy()])
    y = mindspore.Tensor(y.asnumpy()[valid.asnumpy()])

    '''
    mindspore.context.set_context(device_target="CPU")
    print(mindspore.context.get_context("device_target"))
    print(type(x)==mindspore.Tensor)
    print(type(x)==mindspore.Parameter)
    index_batch=ops.MaskedSelect()(index_batch, valid)
    x=ops.MaskedSelect()(x, valid)
    y=ops.MaskedSelect()(y, valid)
    mindspore.
    '''

    uv = dp_uv[index_batch, :, y, x]
    num_pixel = uv.shape[0]

    uv = uv * (uv_res - 1)
    m_round = ops.Round()
    uv_round = m_round(uv).astype("int64").clip(xmin=0, xmax=uv_res - 1)

    index_uv = (uv_round[:, 1] * uv_res + uv_round[:, 0]).copy() + index_batch * uv_res * uv_res

    sampled_feature = feature_map[index_batch, :, y, x]

    y = (2 * y.astype("float32") / (H - 1)) - 1
    x = (2 * x.astype("float32") / (W - 1)) - 1
    concat = ops.Concat(-1)  # dim=1
    sampled_feature = concat([sampled_feature, x[:, None], y[:, None]])

    zero = ops.Zeros()
    warped_w = zero((B * uv_res * uv_res, 1), sampled_feature.dtype)
    index_add = ops.IndexAdd(axis=0)
    warped_w = index_add(warped_w, index_uv.astype("int32"), zero((num_pixel, 1), sampled_feature.dtype))

    warped_feature = zero((B * uv_res * uv_res, C + 2), sampled_feature.dtype)
    warped_feature = index_add(warped_feature, index_uv.astype("int32"), sampled_feature)

    warped_feature = warped_feature / (warped_w + 1e-8)
    warped_feature = concat([warped_feature, (warped_w > 0).astype("float32")])
    warped_feature = warped_feature.reshape(B, uv_res, uv_res, C + 3).transpose(0, 3, 1, 2)

    return warped_feature


# DPNet:returns densepose result
class DPNet(nn.Cell):
    def __init__(self, warp_lv=2, norm_type="BN"):
        super(DPNet, self).__init__()
        nl_layer = nn.ReLU()
        self.warp_lv = warp_lv

        # image encoder
        self.resnet = resnet50backbone(pretrained=True)

        dp_layers = []

        channel_list = [3, 64, 256, 512, 1024, 2048]
        for i in range(warp_lv, 5):
            in_channels = channel_list[i + 1]
            out_channels = channel_list[i]

            dp_layers.append(
                nn.SequentialCell(
                    # nn.ResizeBilinear(),
                    ResizeBilinear(scale_factor=2),
                    ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nl_layer,
                                   norm_type=norm_type)
                )
            )
        self.dp_layers = nn.CellList(dp_layers)

        self.dp_uv_end = nn.SequentialCell(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer, norm_type=norm_type),
                                           nn.Conv2d(32, 2, kernel_size=1),
                                           nn.Sigmoid()
                                           )
        self.dp_mask_end = nn.SequentialCell(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer, norm_type=norm_type),
                                             nn.Conv2d(32, 1, kernel_size=1),
                                             nn.Sigmoid()
                                             )

    def construct(self, image, UV=None):

        codes, features = self.resnet(image)

        dp_feature = features[-1]
        for i in range(len(self.dp_layers) - 1, -1, -1):
            '''
            if isinstance(self.dp_layers[i][0],nn.ResizeBilinear):
                self.dp_layers[i][0](scale_factor=2)
            '''

            dp_feature = self.dp_layers[i](dp_feature)
            dp_feature = dp_feature + features[i - 1 + len(features) - len(self.dp_layers)]
        dp_uv = self.dp_uv_end(dp_feature)
        dp_mask = self.dp_mask_end(dp_feature)
        ops_cat = ops.Concat(1)
        dp_out = ops_cat((dp_mask, dp_uv))

        return dp_out, dp_feature, codes


def get_LNet(options):
    if options.model == "DecoMR":
        uv_net = UVNet(uv_channels=options.uv_channels,
                       uv_res=options.uv_res,
                       warp_lv=options.warp_level,
                       uv_type=options.uv_type,
                       norm_type=options.norm_type
                       )
    return uv_net


# UVNet returns location map
class UVNet(nn.Cell):
    def __init__(self, uv_channels=64, uv_res=128, warp_lv=2, uv_type="SMPL", norm_type="BN"):
        super(UVNet, self).__init__()

        nl_layer = nn.ReLU()

        self.fc_head = nn.SequentialCell(
            nn.Dense(2048, 512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Dense(512, 256)
        )
        self.camera = nn.SequentialCell(
            nn.Dense(2048, 512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dense(256, 3)
        )

        self.warp_lv = warp_lv
        channel_list = [3, 64, 256, 512, 1024, 2048]
        warp_channel = channel_list[warp_lv]
        self.uv_res = uv_res
        self.warp_res = int(256 // (2 ** self.warp_lv))

        if uv_type == "SMPL":
            ref_file = 'models/data/SMPL_ref_map_{0}.npy'.format(
                self.warp_res)
        elif uv_type == 'BF':
            ref_file = 'models/data/BF_ref_map_{0}.npy'.format(
                self.warp_res)

        if not os.path.exists(ref_file):
            sampler = Index_UV_Generator(UV_height=self.warp_res, uv_type=uv_type)
            ref_vert, _ = read_obj('models/data/reference_mesh.obj')
            ref_map = sampler.get_UV_map(mindspore.Tensor(ref_vert, dtype=mindspore.float32))
            np.save(ref_file, ref_map.asnumpy())
        self.ref_map = Tensor(np.load(ref_file), dtype=mindspore.float32).transpose(0, 3, 1, 2)

        self.uv_conv1 = nn.SequentialCell(
            nn.Conv2d(256 + warp_channel + 3 + 3, 2 * warp_channel, kernel_size=1),
            nl_layer,
            nn.Conv2d(2 * warp_channel, 2 * warp_channel, kernel_size=1),
            nl_layer,
            nn.Conv2d(2 * warp_channel, warp_channel, kernel_size=1)
        )

        uv_lv = 0 if uv_res == 256 else 1
        self.hg = HgNet(in_channels=warp_channel, level=5 - warp_lv, nl_layer=nl_layer, norm_type=norm_type)

        cur = min(8, 2 ** (warp_lv - uv_lv))
        prev = cur
        self.uv_conv2 = ConvBottleNeck(warp_channel, uv_channels * cur, nl_layer, norm_type=norm_type)

        layers = []
        for lv in range(warp_lv, uv_lv, -1):
            cur = min(prev, 2 ** (lv - uv_lv - 1))
            layers.append(
                nn.SequentialCell(ResizeBilinear(scale_factor=2),
                                  ConvBottleNeck(uv_channels * prev, uv_channels * cur, nl_layer, norm_type=norm_type)
                                  )
            )
            prev = cur

        self.decoder = nn.SequentialCell(layers)
        self.uv_end = nn.SequentialCell(ConvBottleNeck(uv_channels, 32, nl_layer, norm_type=norm_type),
                                        nn.Conv2d(32, 3, kernel_size=1)
                                        )
        self.concat = ops.Concat(1)

    def construct(self, dp_out, dp_feature, codes):

        n_batch = dp_out.shape[0]

        local_feature = warp_feature(dp_out, dp_feature,
                                     self.warp_res)  # mindspore.Tensor(np.random.random((3, 259, 64, 64)),dtype=mindspore.float32)

        global_feature = self.fc_head(codes)

        global_feature = global_feature[:, :, None, None]  # .repeat(self.warp_res, 2).repeat(self.warp_res,3)
        concat = ops.Concat(2)
        global_feature = concat([global_feature] * self.warp_res)
        concat = ops.Concat(3)
        global_feature = concat([global_feature] * self.warp_res)
        self.ref_map = self.ref_map.astype(local_feature.dtype)

        uv_map = self.concat((local_feature, global_feature, self.ref_map.repeat(n_batch, 0)))

        uv_map = self.uv_conv1(uv_map)
        uv_map = self.hg(uv_map)
        uv_map = self.uv_conv2(uv_map)
        uv_map = self.decoder(uv_map)
        uv_map = self.uv_end(uv_map).transpose(0, 2, 3, 1)

        cam = self.camera(codes)

        return uv_map, cam
