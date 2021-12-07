import argparse
from types import FunctionType
import torch
import torch.nn as nn
import numpy as np # type: ignore
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import matplotlib.pyplot as plt # type: ignore

from src.config import *


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from-where", type=str, default=RESTORE_FROM_WHERE,
                        help="Where restore model parameters from pretrained or saved.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
#    parser.add_argument("--cpu", action="store_true", help="choose to use cpu device.")
#    parser.add_argument("--cpu", default=USE_CPU, help="choose to use cpu device.")
    parser.add_argument("--device", default=DEVICE,
                        help="CPU or GPU")
    parser.add_argument("--tensorboard", action="store_true", help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--dataroot", type=str, default=DATA_PATH,
                        help="Path to the file listing the data.")
    return parser.parse_args()
    #end parse



def lr_poly(base_lr, iter_, max_iter, power):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter, num_steps, args, times=1):
    lr = lr_poly(args.learning_rate, i_iter, num_steps, args.power)
    optimizer.param_groups[0]["lr"] = lr * times




def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1

def create_pascal_label_colormap(class_num):
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((class_num, 3), dtype=int)
    ind = np.arange(class_num, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


n_classes = 20
color = create_pascal_label_colormap(n_classes)

def to_color_img(img):
    #下面的0代表batch的第0个元素
    score_i = img[0,...]
    score_i = score_i.cpu().numpy()
    #转换通道
    score_i = np.transpose(score_i,(1,2,0))
    # one hot转一个channel
    score_i = np.argmax(score_i,axis=2)
    #color为上面生成的color list
    color_img = color[score_i]
    return color_img


"""
print_args : (args: Namespace) -> Void
"""
def print_args(args):
    dic = vars(args)
    for k in dic:
        log("    {}: {}".format(k, dic[k]))
        #end for
    #end print_args

'''
print_config : (args: Namespace) -> Void
'''
def print_config(args):
    log("Program configurations:")
    print_args(args)
    log('End of configurations.')
    print()
    #end print_config


#def mIoU(pred, label):
def confusion_matrix(a, b, n):
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    #print(np.unique(a[k]), np.unique(b[k]))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))



def show_sample(batch):
    imgs, msks = batch['image'], batch['label']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].imshow(imgs[0].permute(1, 2, 0))
    axs[1].imshow(msks[0], cmap = 'tab20', vmin = 0, vmax = 21)
    #axs.set_title("test")
    #axs.grid(True)

    log("Displaying image of {}".format(batch['name']))
    # plt.colorbar()
    plt.show()



# Logging functions

def prefixed_with(msg: str, prefix: str) -> str:
    return '{} {}'.format(prefix, msg)

def blank_line(n: int = 1) -> None:
    for _ in range(0, n): print()

def log(msg: str) -> None:
    prefix = '[info]'
    log_msg = prefixed_with(msg, prefix)
    print(log_msg)

def debug(msg: str, description = "") -> None:
    prefix = '[debug]'
    description = '' if description == '' else description + ': '
    debug_msg = '{} {}{}'.format(prefix, description, msg)
    print(debug_msg)

def custom_log(prefix: str) -> FunctionType:
    return lambda msg: prefixed_with(msg, prefix)




class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # self.ce = nn.BCELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, input, target):
        logp = self.ce(input, target)
        # print(f'logp:{logp}')
        p = torch.exp(-logp)
        # print(f'p:{p}')
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()
