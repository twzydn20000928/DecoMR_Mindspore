from utils.train_options import TrainOptions
from train.trainer3 import Trainer3
import mindspore.context as context
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)
#context.set_context(pynative_synchronize=True)
                                                                                     
print(context.get_context("device_id"))

if __name__ == '__main__':
    print("进行号", os.getppid())
    options = TrainOptions().parse_args()
    print("option == ", options)
    trainer = Trainer3(options)
    # 
    trainer.train()
