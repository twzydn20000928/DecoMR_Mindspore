import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import numpy as ms_np
import mindspore.nn as nn


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    zero = Tensor(np.zeros([]), mindspore.float32)
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    shape = P.Shape()
    num_batch = shape(theta)[0]
    x = np.linspace(-1.0, 1.0, height)
    y = np.linspace(-1.0, 1.0, width)
    x_t, y_t = np.meshgrid(x, y)
    x_t = Tensor(x_t, mindspore.float32)
    y_t = Tensor(y_t, mindspore.float32)
    expand_dims = P.ExpandDims()
    x_t = expand_dims(x_t, 0)
    y_t = expand_dims(y_t, 0)
    flatten = P.Flatten()
    x_t_flat = flatten(x_t)
    y_t_flat = flatten(y_t)
    oneslike = P.OnesLike()
    ones = oneslike(x_t_flat)
    concat = P.Concat()
    sampling_grid = concat((x_t_flat, y_t_flat, ones))
    sampling_grid = expand_dims(sampling_grid, 0)


    cast = P.Cast()
    theta = cast(theta, mindspore.float32)

    # transform the sampling grid - batch multiply
    tile = P.Tile()
    sampling_grid = tile(sampling_grid, (num_batch, 1, 1))
    cast = P.Cast()
    sampling_grid = cast(sampling_grid, mindspore.float32)

    matmul = P.BatchMatMul()
    batch_grids = matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    reshape = P.Reshape()
    batch_grids = reshape(batch_grids, (num_batch, 2, height, width))
    return batch_grids

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = P.Shape()
    img_shape = shape(x)
    batch_size = img_shape[0]
    height = img_shape[1]
    width = img_shape[2]
    #img[:, 0, :, :] = self.zero
    #img[:, height - 1, :, :] = self.zero
    #img[:, :, 0, :] = self.zero
    #img[:, :, width - 1, :] = self.zero
    batch_idx = ms_np.arange(0, batch_size)
    batch_idx = batch_idx.reshape((batch_size, 1, 1))
    tile = P.Tile()
    b = tile(batch_idx, (1, height, width))
    cast = P.Cast()
    b = cast(b,mindspore.float32)

    #batch_idx = P.Slice()(self.batch_idx, (0, 0, 0), (batch_size, 1, 1))

    expand_dims = P.ExpandDims()
    b = expand_dims(b, 3)
    x = expand_dims(x, 3)
    y = expand_dims(y, 3)

    concat = P.Concat(3)
    indices = concat((b, y, x))

    indices = cast(indices, mindspore.int32)
    gather_nd = P.GatherNd()

    return cast(gather_nd(img, indices), mindspore.float32)

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    shape = P.Shape()
    H = shape(img)[1]
    W = shape(img)[2]
    cast = P.Cast()
    max_y = cast(H - 1, mindspore.float32)
    max_x = cast(W - 1, mindspore.float32)
    zeros = P.Zeros()
    zero = zeros((),mindspore.float32)

    # rescale x and y to [0, W-1/H-1]
    x = 0.5 * ((x + 1.0) * (max_x - 1))
    y = 0.5 * ((y + 1.0) * (max_y - 1))

    # grab 4 nearest corner points for each (x_i, y_i)
    floor = P.Floor()
    x0 = floor(x)
    x1 = x0 + 1
    y0 = floor(y)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = C.clip_by_value(x0, zero, max_x)
    x1 = C.clip_by_value(x1, zero, max_x)
    y0 = C.clip_by_value(y0, zero, max_y)
    y1 = C.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = cast(x0, mindspore.float32)
    x1 = cast(x1, mindspore.float32)
    y0 = cast(y0, mindspore.float32)
    y1 = cast(y1, mindspore.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    expand_dims = P.ExpandDims()
    wa = expand_dims(wa, 3)
    wb = expand_dims(wb, 3)
    wc = expand_dims(wc, 3)
    wd = expand_dims(wd, 3)

    # compute output
    add_n = P.AddN()
    out = add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out
