import os
import os.path as osp
import time
import datetime
import cv2
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssim

from models import get_model
from utils.data import *
from utils.lr_scheduler import *
from utils.metrics import psnr
from utils.logger import setup_logger


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Compressive Sensing CT-Net')

    #
    # Dataset parameters
    #
    parser.add_argument('--data_dir', type=str, default='../datasets',
                        help='train data dir')
    parser.add_argument('--pkl-path', type=str,
                        default=r'../checkpoints/ctnet_cs25_stage9_hemlayer6_hemchannel32.pkl',
                        help='checkpoint file')

    #
    # Training parameters
    #
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    #
    # Net parameters
    #
    parser.add_argument('--model-name', type=str, default='ctnet',
                        help='model name')
    parser.add_argument('--cs-ratio', type=int, default=25,
                        choices=[10, 25, 30, 40, 50], help='CS ratio')
    parser.add_argument('--stage-num', type=int, default=9,
                        help='stage number')
    parser.add_argument('--hem-layers', type=int, default=6,
                        help='hem layers')
    parser.add_argument('--hem-channel', type=int, default=32,
                        help='hem channel')
    parser.add_argument('--block-size', type=int, default=33,
                        help='block size')

    args = parser.parse_args()

    # seed
    if not args.seed == 0:
        set_seeds(args.seed)

    # val
    args.val_dict = {
        'set11': osp.join(args.data_dir, 'Set11'),
        'bsd68': osp.join(args.data_dir, 'CBSD68'),
        'urban100': osp.join(args.data_dir, 'Urban100')
    }

    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0
        self.ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

        ## GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## model
        net_kwargs = {'stage_num': args.stage_num, 'n_input': self.ratio_dict[args.cs_ratio],
                      'hem_layers': args.hem_layers, 'hem_channel': args.hem_channel}
        self.net = get_model(args.model_name, **net_kwargs).to(self.device)
        self.net.load_state_dict(torch.load(args.pkl_path, map_location=self.device))

        ## val
        val_img_paths = {}
        for key in args.val_dict:
            data_path = args.val_dict[key]
            if not os.path.exists(data_path):
                err_str = '{} path not exist'.format(data_path)
                raise ValueError(err_str)
            val_img_paths[key] = []
            for filename in os.listdir(data_path):
                name, suffix = osp.splitext(filename)
                if suffix in ['.png', '.jpg', '.tif']:
                    val_img_paths[key].append(osp.join(data_path, filename))
        self.val_img_paths = val_img_paths


    def validation(self):
        self.net.eval()
        base_log = "Data: {:s}, PSNR: {:.4f}, SSIM: {:.4f}"
        for data in self.val_img_paths:
            psnrs, ssims = [], []
            for path in self.val_img_paths[data]:
                rec_psnr, rec_ssim, rec_img = self.val_single_img(path)
                psnrs.append(rec_psnr)
                ssims.append(rec_ssim)

            print(base_log.format(data, np.mean(psnrs), np.mean(ssims)))


    def val_single_img(self, img_path):
        img = cv2.imread(img_path, 1)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_rec_yuv = img_yuv.copy()

        Iorg_y = img_yuv[:, :, 0]

        # pad
        row, col = Iorg_y.shape
        row_pad = args.block_size - np.mod(row, args.block_size)
        col_pad = args.block_size - np.mod(col, args.block_size)
        Ipad = cv2.copyMakeBorder(Iorg_y, 0, row_pad, 0, col_pad, borderType=cv2.BORDER_CONSTANT, value=0)
        row_new, col_new = Ipad.shape

        Img_output = Ipad.reshape(1, 1, row_new, col_new) / 255.0

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(self.device)

        with torch.no_grad():
            x_output, loss_phi = self.net(batch_x)

        pred = x_output.cpu().data.numpy().squeeze()
        X_rec = np.clip(pred[:row, :col], 0, 1).astype(np.float64)

        rec_PSNR = psnr(X_rec * 255, Iorg_y.astype(np.float64))
        rec_SSIM = ssim(X_rec * 255, Iorg_y.astype(np.float64), data_range=255)

        # reco
        img_rec_yuv[:, :, 0] = X_rec * 255

        im_rec_rgb = cv2.cvtColor(img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        return rec_PSNR, rec_SSIM, im_rec_rgb


if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    trainer.validation()





