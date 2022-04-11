import torch
from mindspore import Tensor, save_checkpoint

default_file = '/data/users/user1/Master_04/Deco_MR_mindspore/models/resnet50-19c8e357.pth'
# read pth file
par_dict = torch.load(default_file)
params_list = []
for name in par_dict:
    param_dict = {}
    parameter = par_dict[name]
    if name[-19:] == 'downsample.0.weight':
        param_dict['name'] = name
    elif name[0:3] == 'fc.':
        param_dict['name'] = name
    elif name[0:4] == 'conv':
        param_dict['name'] = name
    elif name[9:13] == 'conv':
        param_dict['name'] = name
    #bn
    elif name[-12:] == 'running_mean':
        param_dict['name'] = name[:-12] + 'moving_mean'
    elif name[-11:] == 'running_var':
        param_dict['name'] = name[:-11] + 'moving_variance'
    elif name[-6:] == 'weight':
        param_dict['name'] = name[:-6] + 'gamma'
    elif name[-4:] == 'bias':
        param_dict['name'] = name[:-4] + 'beta'

    param_dict['data'] = Tensor(parameter.data.numpy())
    params_list.append(param_dict)

save_checkpoint(params_list,  'ms_resnet50.ckpt')