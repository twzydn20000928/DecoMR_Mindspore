from utils import TrainOptions
from train import Trainer
import mindspore.context as context

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
print(context.get_context("device_target"))

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    print("option == ", options)
    trainer = Trainer(options)
    # 
    trainer.train()
