""""
This file includes the full training procedure.
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")
# from torch.nn.parallel import data_parallel
from train.base_trainer import BaseTrainer
# from datasets import create_dataset
# 更换为了torch处理数据
from datasets.datasets_pytorch import create_dataset
from models import SMPL
from models.dense_cnn import DPNet, get_LNet
from models.uv_generator import Index_UV_Generator, cal_uv_weight
import mindspore.dataset as ds
import os
import utils.config as cfg
from train.myTrainOneStepCell import *
from train.myTrainOneStepEndCell import *
from train.myWithLossCell import *
from train.myWithLossEndCell import *
from train.loss import *
import mindspore.context as context


class Trainer(BaseTrainer):
    def init_fn(self):
        # create training dataset
        self.train_ds = create_dataset(self.options.dataset, self.options, use_IUV=True)

        self.dp_res = int(self.options.img_res // (2 ** self.options.warp_level))

        CNet = DPNet(warp_lv=self.options.warp_level,
                     norm_type=self.options.norm_type)  # .to(self.device)

        LNet = get_LNet(self.options)  # .to(self.device)
        self.smpl = SMPL()  # .to(self.device)
        self.female_smpl = SMPL(cfg.FEMALE_SMPL_FILE)  # .to(self.device)
        self.male_smpl = SMPL(cfg.MALE_SMPL_FILE)  # .to(self.device)
        # self.female_smpl = SMPL()#.to(self.device)
        # self.male_smpl = SMPL()#.to(self.device)

        uv_res = self.options.uv_res
        self.uv_type = self.options.uv_type
        self.sampler = Index_UV_Generator(UV_height=uv_res, UV_width=-1, uv_type=self.uv_type)  # .to(self.device)

        weight_file = 'models/data/weight_p24_h{:04d}_w{:04d}_{}.npy'.format(uv_res, uv_res,
                                                                                                         self.uv_type)
        if not os.path.exists(weight_file):
            cal_uv_weight(self.sampler, weight_file)

        uv_weight = mindspore.Tensor(np.load(weight_file), mindspore.float32)  # .to(self.device).float()
        uv_weight = uv_weight * self.sampler.mask.astype("float32")  # .to(uv_weight.device)
        uv_weight = uv_weight / uv_weight.mean()
        self.uv_weight = uv_weight[None, :, :, None]
        self.tv_factor = (uv_res - 1) * (uv_res - 1)

        # Setup an optimizer
        if self.options.stage == 'dp':
            self.optimizer = nn.Adam(
                params=list(CNet.trainable_params()),
                learning_rate=self.options.lr,
                beta1=self.options.adam_beta1,
                weight_decay=self.options.wd)
            self.models_dict = {'CNet': CNet}
            self.optimizers_dict = {'optimizer': self.optimizer}

        else:
            self.optimizer = nn.Adam(
                params=list(CNet.trainable_params()) + list(LNet.trainable_params()),
                learning_rate=self.options.lr,
                beta1=self.options.adam_beta1,
                weight_decay=self.options.wd)
            self.models_dict = {'CNet': CNet, 'LNet': LNet}
            self.optimizers_dict = {'optimizer': self.optimizer}

        # Create loss functions
        self.criterion_shape = nn.L1Loss()  # .to(self.device)
        self.criterion_uv = nn.L1Loss()  # .to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none')  # .to(self.device)
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none')  # .to(self.device)
        self.criterion_regr = nn.MSELoss()  # .to(self.device)

        # LSP indices from full list of keypoints
        self.to_lsp = list(range(14))
        # self.renderer = Renderer(faces=self.smpl.faces.asnumpy())#.cpu().numpy()

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        # if self.options.pretrained_checkpoint is not None:
        # print(self.options.pretrained_checkpoint)
        # self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        '''
        if self.options.stage=="dp":
            loss=Loss(self.options)
            self.wlc=WithLossCell(CNet,self,self.options)
            self.net_dp=TrainOneStepCell(self.wlc,self.optimizer)
        
        else:
            self.wlec=WithLossEndCell(CNet,LNet,self,self.options)
            self.net_end=TrainOneStepEndCell(self.wlec,self.optimizer)
        '''
        self.wlc = WithLossCell(CNet, self, self.options)
        self.net_dp = TrainOneStepCell(self.wlc, self.optimizer)

        self.wlec = WithLossEndCell(CNet, LNet, self, self.options)
        self.net_end = TrainOneStepEndCell(self.wlec, self.optimizer)

        self.total_loss = 0

    def train_step(self, input_batch):
        """Training step."""
        dtype = mindspore.float32
        # print(input_batch['img'].dtype)
        if self.options.stage == 'dp':
            self.net_dp.set_train(True)
            # Grab data from the batch
            has_dp = input_batch['has_dp']
            images = input_batch['img']
            # print(images.shape)
            gt_dp_iuv = input_batch['gt_iuv']

            img_orig = input_batch['img_orig']
            gt_dp_iuv_temp = gt_dp_iuv.asnumpy()
            gt_dp_iuv_temp[:, 1:] = gt_dp_iuv.asnumpy()[:, 1:] / 255.0
            gt_dp_iuv = Tensor(gt_dp_iuv_temp, mindspore.float32)

            batch_size = images.shape[0]

            if self.options.adaptive_weight:
                fit_joint_error = input_batch['fit_joint_error']
                ada_weight = self.error_adaptive_weight(fit_joint_error).astype(dtype)
            else:
                # ada_weight = pred_scale.new_ones(batch_size).type(dtype)
                ada_weight = None

            loss_total = self.net_dp(images, gt_dp_iuv, img_orig, has_dp, ada_weight, self.step_count)

            # for visualize
            if (self.step_count + 1) % self.options.summary_steps == 0:
                self.vis_data = self.wlc.get_data()

            # Pack output arguments to be used for visualization in a list
            losses = self.wlc.get_losses()
            out_args = {key: losses[key].copy().asnumpy() for key in losses.keys()}  # detach().item()
            out_args['total'] = loss_total.copy().asnumpy()  # .detach().item()
            self.loss_item = out_args
            self.total_loss += loss_total

        elif self.options.stage == 'end':

            self.net_end.set_train(True)

            # Grab data from the batch
            # gt_keypoints_2d = input_batch['keypoints']
            # gt_keypoints_3d = input_batch['pose_3d']
            # gt_keypoints_2d = torch.cat([input_batch['keypoints'], input_batch['keypoints_smpl']], dim=1)
            # gt_keypoints_3d = torch.cat([input_batch['pose_3d'], input_batch['pose_3d_smpl']], dim=1)
            gt_keypoints_2d = input_batch['keypoints']
            gt_keypoints_3d = input_batch['pose_3d']
            has_pose_3d = input_batch['has_pose_3d']

            gt_keypoints_2d_smpl = input_batch['keypoints_smpl']
            gt_keypoints_3d_smpl = input_batch['pose_3d_smpl']
            has_pose_3d_smpl = input_batch['has_pose_3d_smpl']

            gt_pose = input_batch['pose']
            gt_betas = input_batch['betas']
            has_smpl = input_batch['has_smpl']
            has_dp = input_batch['has_dp']
            images = input_batch['img']
            img_orig = input_batch['img_orig']
            gender = input_batch['gender']

            gt_dp_iuv = input_batch['gt_iuv']
            gt_dp_iuv[:, 1:] = gt_dp_iuv[:, 1:] / 255.0
            batch_size = images.shape[0]

            gt_vertices = ops.Zeros()((batch_size, 6890, 3), images.dtype)  # to.device
            '''
            if images.is_cuda and self.options.ngpu > 1:
                with torch.no_grad():
                    gt_vertices.asnumpy()[gender.asnumpy() < 0] = data_parallel(self.smpl, (Tensor(gt_pose.asnumpy()[gender.asnumpy() < 0]), Tensor(gt_betas.asnumpy()[gender.asnumpy() < 0])), range(self.options.ngpu))
                    gt_vertices.asnumpy()[gender.asnumpy() == 0]= data_parallel(self.male_smpl, (Tensor(gt_pose.asnumpy()[gender.asnumpy() == 0]), Tensor(gt_betas.asnumpy()[gender.asnumpy() == 0])), range(self.options.ngpu))
                    gt_vertices.asnumpy()[gender.asnumpy() == 1]= data_parallel(self.female_smpl, (Tensor(gt_pose.asnumpy()[gender.asnumpy() == 1], gt_betas.asnumpy()[gender.asnumpy() == 1]), range(self.options.ngpu))
                    gt_vertices=Tensor(gt_vertices)
                    gt_uv_map = data_parallel(self.sampler, gt_vertices, range(self.options.ngpu))
                pred_dp, dp_feature, codes = data_parallel(self.CNet, images, range(self.options.ngpu))
                pred_uv_map, pred_camera = data_parallel(self.LNet, (pred_dp, dp_feature, codes),
                                                                         range(self.options.ngpu))
            else:
                # gt_vertices = self.smpl(gt_pose, gt_betas)
                with torch.no_grad():
                    gt_vertices.asnumpy()[gender.asnumpy() < 0] = self.smpl(Tensor(gt_pose.asnumpy()[gender.asnumpy() < 0]), Tensor(gt_betas.asnumpy()[gender.asnumpy() < 0]))
                    gt_vertices.asnumpy()[gender.asnumpy() == 0]= self.male_smpl(Tensor(gt_pose.asnumpy()[gender.asnumpy() == 0], Tensor(gt_betas.asnumpy()[gender.asnumpy() == 0]))
                    gt_vertices.asnumpy()[gender.asnumpy() == 1] = self.female_smpl(Tensor(gt_pose.asnumpy()[gender.asnumpy() == 1]), Tensor(gt_betas.asnumpy()[gender.asnumpy() == 1]))
                    gt_vertices=Tensor(gt_vertices)
                    gt_uv_map = self.sampler.get_UV_map(gt_vertices.float())
                
            '''
            gt_vertices.asnumpy()[gender.asnumpy() < 0] = self.smpl(Tensor(gt_pose.asnumpy()[gender.asnumpy() < 0]),
                                                                    Tensor(gt_betas.asnumpy()[
                                                                               gender.asnumpy() < 0])).asnumpy()

            x0 = gt_pose.asnumpy()[gender.asnumpy() == 0]
            x1 = gt_pose.asnumpy()[gender.asnumpy() == 1]
            if x0.shape[0] != 0:
                gt_vertices.asnumpy()[gender.asnumpy() == 0] = self.male_smpl(
                    Tensor(gt_pose.asnumpy()[gender.asnumpy() == 0]),
                    Tensor(gt_betas.asnumpy()[gender.asnumpy() == 0])).asnumpy()
            if x1.shape[0] != 0:
                gt_vertices.asnumpy()[gender.asnumpy() == 1] = self.female_smpl(
                    Tensor(gt_pose.asnumpy()[gender.asnumpy() == 1]),
                    Tensor(gt_betas.asnumpy()[gender.asnumpy() == 1])).asnumpy()

            # gt_vertices.asnumpy()[gender.asnumpy() == 0]= self.male_smpl(Tensor(gt_pose.asnumpy()[gender.asnumpy() == 0]), Tensor(gt_betas.asnumpy()[gender.asnumpy() == 0])).asnumpy()
            # gt_vertices.asnumpy()[gender.asnumpy() == 1] = self.female_smpl(Tensor(gt_pose.asnumpy()[gender.asnumpy() == 1]), Tensor(gt_betas.asnumpy()[gender.asnumpy() == 1])).asnumpy()
            gt_vertices = Tensor(gt_vertices, mindspore.float32)
            gt_uv_map = self.sampler.get_UV_map(gt_vertices.astype("float32"))
            # gt_uv_map=mindspore.Tensor(np.random.random((100,128,128,3)),mindspore.float32)

            if self.options.adaptive_weight:
                # Get the confidence of the GT mesh, which is used as the weight of loss item.
                # The confidence is related to the fitting error and for the data with GT SMPL parameters,
                # the confidence is 1.0
                fit_joint_error = input_batch['fit_joint_error']
                ada_weight = self.error_adaptive_weight(fit_joint_error).astype(dtype)
            else:
                ada_weight = None
            time1 = time.time()
            loss_total = self.net_end(images, img_orig, gt_uv_map, gt_keypoints_2d, gt_keypoints_3d, has_pose_3d,
                                      gt_keypoints_2d_smpl, gt_keypoints_3d_smpl, has_pose_3d_smpl, gt_pose, gt_betas,
                                      has_smpl, has_dp, gender, gt_dp_iuv, batch_size, ada_weight, self.step_count)
            time2 = time.time()
            print("model", time2 - time1)
            print("=============================")

            # for visualize
            if (self.step_count + 1) % self.options.summary_steps == 0:
                self.vis_data = self.wlec.data

            # Pack output arguments to be used for visualization in a list
            losses = self.wlec.get_losses()
            out_args = {key: losses[key].copy().asnumpy() for key in losses.keys()}
            out_args['total'] = loss_total.copy().asnumpy()
            self.loss_item = out_args
            self.total_loss += loss_total
        return out_args

    def train_summaries(self, batch, epoch):
        """Tensorboard logging."""
        if self.options.stage == 'dp':
            dtype = self.vis_data['pred_dp'].dtype
            rend_imgs = []
            vis_size = self.vis_data['pred_dp'].shape[0]
            # Do visualization for the first 4 images of the batch
            for i in range(vis_size):
                
                img = self.vis_data['image'][i].asnumpy().transpose(1, 2, 0)
                H, W, C = img.shape
                rend_img = img.transpose(2, 0, 1)

                gt_dp = self.vis_data['gt_dp'][i]
                resize = ResizeNearestNeighbor((H, W))
                gt_dp = resize(gt_dp[None, :])[0]
                gt_dp = gt_dp.asnumpy()
                rend_img = np.concatenate((rend_img, gt_dp), axis=2)
                pred_dp = self.vis_data['pred_dp'][i]
                pred_dp[0] = Tensor((pred_dp[0] > 0.5), dtype)
                pred_dp[1:] = pred_dp[1:] * pred_dp[0]
                resize = ResizeNearestNeighbor((H, W))
                gt_dp = resize(gt_dp[None, :])[0]
                pred_dp = pred_dp.asnumpy()
                rend_img = np.concatenate((rend_img, pred_dp), axis=2)
                rend_imgs.append(Tensor(rend_img))
            rend_imgs = make_grid(rend_imgs, nrow=1)  # make_grid不影响运行，算子未找到
            self.summary_writer.add_image('imgs', rend_imgs, self.step_count)

        else:
            gt_keypoints_2d = self.vis_data['gt_joint'].asnumpy()
            pred_vertices = self.vis_data['pred_vert']
            pred_keypoints_2d = self.vis_data['pred_joint']
            pred_camera = self.vis_data['pred_cam']
            dtype = pred_camera.dtype
            rend_imgs = []
            vis_size = pred_vertices.shape[0]
            # Do visualization for the first 4 images of the batch
            for i in range(vis_size):
                
                img = self.vis_data['image'][i].asnumpy().transpose(1, 2, 0)
                H, W, C = img.shape

                # Get LSP keypoints from the full list of keypoints
                gt_keypoints_2d_ = gt_keypoints_2d[i, self.to_lsp]
                pred_keypoints_2d_ = pred_keypoints_2d.asnumpy()[i, self.to_lsp]
                vertices = pred_vertices[i].asnumpy()
                cam = pred_camera[i].asnumpy()
                # Visualize reconstruction and detected pose
                rend_img = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices,
                                                    pred_keypoints_2d_, cam, self.renderer)
                rend_img = rend_img.transpose(2, 0, 1)

                if 'gt_vert' in self.vis_data.keys():
                    rend_img2 = vis_mesh(img, self.vis_data['gt_vert'][i].asnumpy(), cam, self.renderer, color='blue')
                    rend_img2 = rend_img2.transpose(2, 0, 1)
                    rend_img = np.concatenate((rend_img, rend_img2), axis=2)

                gt_dp = self.vis_data['gt_dp'][i]
                resize = ResizeNearestNeighbor((H, W))
                gt_dp = resize(gt_dp[None, :])[0]
                gt_dp = gt_dp.asnumpy()
                rend_img = np.concatenate((rend_img, gt_dp), axis=2)
                pred_dp = self.vis_data['pred_dp'][i]
                pred_dp[0] = (pred_dp[0] > 0.5)
                pred_dp[1:] = pred_dp[1:] * pred_dp[0]
                resize = ResizeNearestNeighbor((H, W))
                pred_dp = resize(pred_dp[None, :])[0]
                pred_dp = pred_dp.asnumpy()
                rend_img = np.concatenate((rend_img, pred_dp), axis=2)

                # import matplotlib.pyplot as plt
                # plt.imshow(rend_img.transpose([1, 2, 0]))
                rend_imgs.append(torch.from_numpy(rend_img))

            rend_imgs = make_grid(rend_imgs, nrow=1)  # make_grid不影响运行

            uv_maps = []
            for i in range(vis_size):
                uv_temp = torch.cat((self.vis_data['pred_uv'][i], self.vis_data['gt_uv'][i]), dim=1)
                uv_maps.append(uv_temp.permute(2, 0, 1))

            uv_maps = make_grid(uv_maps, nrow=1)  # make_grid
            uv_maps = uv_maps.abs()
            uv_maps = uv_maps / uv_maps.max()

            # Save results in Tensorboard
            self.summary_writer.add_image('imgs', rend_imgs, self.step_count)
            self.summary_writer.add_image('uv_maps', uv_maps, self.step_count)

        for key in self.loss_item.keys():
            self.summary_writer.add_scalar('loss_' + key, self.loss_item[key], self.step_count)

    def train(self):
        """Training process."""
        dataset = ds.GeneratorDataset(self.train_ds, [
            "scale",
            "center",
            "orig_shape",
            "img_orig",
            "img",
            "has_smpl",
            "pose",
            "betas",
            "has_pose_3d",
            "pose_3d",
            "keypoints",
            "keypoints_smpl",
            "pose_3d_smpl",
            "has_pose_3d_smpl",
            "gender",
            "gt_iuv",
            "has_dp",
            "fit_joint_error",
            # "imgname"
        ], shuffle=False, num_parallel_workers=1, python_multiprocessing=False)  # self.options.shuffle_train
        dataset = dataset.batch(batch_size=8)
        # Run training for num_epochs epochs

        # for epoch in range(self.epoch_count, self.options.num_epochs):
        #     batch_len = len(self.train_ds) // self.options.batch_size
        #     #dataset=dataset.to_device()
        #     #print(dataset.device_id)
        #     for batch in dataset.create_dict_iterator():
        #          #batch['img'] = batch['img'].transpose(0, 3, 1, 2)
        #          loss_dict = self.train_step(batch)
        #     print("start train!")
        #     #数据类型转换
        #     # batch['img'] = batch['img'].transpose(0, 3, 1, 2)
        #     loss_dict = self.train_step(batch)
        #     print("len(batch) = " , len(batch['img']))
        #     # break
        a = []
        for epoch in range(self.epoch_count, self.options.num_epochs):
            batch_len = len(self.train_ds) // self.options.batch_size
            # dataset=dataset.to_device()
            # print(dataset.device_id)
            epoch_num = 0
            batch_ = {}
            self.total_loss = 0
            '''
            for batch in dataset.create_dict_iterator():
              
                batch_[epoch_num] = batch
                epoch_num += 1
                if epoch_num > 800:   #数据集先 5epoch * 20 训练 ，epoch数量可以自己调
                    break;
            for batch in batch_:
                loss_dict = self.train_step(batch_[batch])
            '''

            for batch in dataset.create_dict_iterator():
                time1 = time.time()
                loss_dict = self.train_step(batch)
                time2 = time.time()
                print("train", time2 - time1)
            print("epoch: %d loss: %f" % (epoch, self.total_loss.asnumpy() / len(batch_)))
            # mindspore.save_checkpoint(self.net_dp, "net_dp.ckpt")

    '''
    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.num_epochs):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            
            
          
            #train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
            #                                         batch_size=self.options.batch_size,
            #                                         num_workers=self.options.num_workers,
            #                                         pin_memory=self.options.pin_memory,
            #                                         shuffle=self.options.shuffle_train)
            #
            #train_data_loader.batch(2)
            #iterator = train_data_loader.create_dict_iterator()
            
            
            dataset = ds.GeneratorDataset(self.train_ds, [
                                                        "scale",
                                                        "center",
                                                        "orig_shape",
                                                        "img_orig",
                                                        "img",
                                                        "has_smpl",
                                                        "pose",
                                                        "betas",
                                                        "has_pose_3d",
                                                        "pose_3d",
                                                        "keypoints",
                                                        "keypoints_smpl",
                                                        "pose_3d_smpl",
                                                        "has_pose_3d_smpl",
                                                        "gender",
                                                        "gt_iuv",
                                                        "has_dp",
                                                        "fit_joint_error",
                                                        # "imgname"
                                                          ], shuffle=self.options.shuffle_train)
          
            #decode_op = c_vision.Decode()
            #normalize_op = c_vision.Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
            #transforms_list = [decode_op, normalize_op]
            #dataset = dataset.map(operations=normalize_op, input_columns=["img"])
            dataset = dataset.batch(batch_size=24)
            batch_len = len(self.train_ds) // self.options.batch_size
            checkpoint = 0
            if self.checkpoint is not None:
            	checkpoint=self.checkpoint["batch_idx"]
           
            	
            #iterator=dataset.create_dict_iterator()
            
            
            #data_stream=tqdm(iterator,desc='Epoch ' + str(epoch),
            #                   total=len(self.train_ds) // self.options.batch_size,
            #                   initial=train_data_loader.checkpoint_batch_idx)
            
            
            for step,batch in enumerate(dataset.create_dict_iterator(), checkpoint):
                if time.time() < self.endtime:
                    print("train_start")
                    #batch = {k: batch[k] if isinstance(batch[k], mindspore.Tensor) else batch[k] for k in batch}#.to(self.device)
                   
                    loss_dict = self.train_step(batch)
                    print("train_end")
                    self.step_count += 1

                    tqdm_info = 'Epoch:%d| %d/%d ' % (epoch, step, batch_len)
                    for k, v in loss_dict.items():
                        tqdm_info += ' %s:%.4f' % (k, v)
                    data_stream.set_description(tqdm_info)

                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(step, epoch)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0 and self.step_count > 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                                   self.options.batch_size, sampler.dataset_perm,
                                                   self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                    

                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                                               self.options.batch_size, sampler.dataset_perm,
                                               self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs just iterate over the dataset as usual
            self.checkpoint = None
            # save checkpoint after each 10 epoch
            if (epoch + 1) % 10 == 0:
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                           self.options.batch_size, None, self.step_count)

        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                   self.options.batch_size, None, self.step_count, checkpoint_filename='final')
        return
      '''
