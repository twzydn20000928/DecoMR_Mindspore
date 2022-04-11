import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np


class WithLossCell(nn.Cell):
    def __init__(self,cnet,criterion,option):
        super(WithLossCell,self).__init__(auto_prefix=False)
        self.CNet=cnet
        self.criterion=criterion
        self.options=option
        self.data=None
        self.losses=None

    
    def get_losses(self):
        return self.losses
    def get_data(self):
        return self.data

    def construct(self,images,gt_dp_iuv,img_orig,has_dp,ada_weight,step_count):
        if ada_weight!=None and ada_weight.ndim==0 and ada_weight.asnumpy()==1.0:
            ada_weight=None
        if type(step_count)==Tensor:
            step_count=None
        	
        '''
        if images.is_cuda and self.options.ngpu > 1:
            pred_dp, dp_feature, codes = data_parallel(self.CNet, images, range(self.options.ngpu))
        else:
            pred_dp, dp_feature, codes = self.CNet(images)
        '''
        pred_dp, dp_feature, codes = self.CNet(images)  
        # print(pred_dp.asnumpy()[0,0,0,0])
       
        
        losses = {}
        '''loss on dense pose result'''
        # gt_dp_iuv_numpy = gt_dp_iuv.copy()
        # gt_dp_iuv_numpy = gt_dp_iuv_numpy.asnumpy()
        loss_dp_mask, loss_dp_uv = self.criterion.dp_loss(pred_dp, gt_dp_iuv, has_dp, ada_weight)
        '''
        with open('logs/gt_dp_iuv.txt', 'a') as f:
            x = gt_dp_iuv.asnumpy()
            x = x.tolist()
            strNums = [str(x_i) for x_i in x]
            str1 = ",".join(strNums)
            f.write(str1)
        '''
        loss_dp_mask = loss_dp_mask * self.options.lam_dp_mask
        loss_dp_uv = loss_dp_uv * self.options.lam_dp_uv
        losses['dp_mask'] = loss_dp_mask
        losses['dp_uv'] = loss_dp_uv
        '''
        with open('logs/dp_loss.txt', 'a') as f:
            all_txt = "dp_mask = " + str(loss_dp_mask.asnumpy()) + "   " + "dp_uv = " + str(
                loss_dp_uv.asnumpy())
            all_txt += "\n"
            f.write(all_txt)
        '''
        loss_total=0
        for loss in losses.values():
            loss_total+=loss
        
        # for visualize
        
        if step_count!=None and (step_count + 1) % self.options.summary_steps == 0:
            data = {}
            vis_num = min(4, batch_size)
            data['image'] = img_orig[0:vis_num].copy()#.detach()
            data['pred_dp'] = pred_dp[0:vis_num].copy()#.detach()
            data['gt_dp'] = gt_dp_iuv[0:vis_num].copy()#.detach()
            self.data = data
        
        self.losses=losses
        #print("loss_total",loss_total)
        return loss_total
        
