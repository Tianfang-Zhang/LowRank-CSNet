import os
import os.path as osp
import time
import datetime
import cv2
from argparse import ArgumentParser
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

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
    parser.add_argument('--data-dir', type=str, default='../datasets/',
                        help='train data dir')
    parser.add_argument('--data-name-33', type=str, default='Training_Data.mat',
                        help='train data name')
    parser.add_argument('--data-name-99', type=str, default='Training_Data_99x99.mat',
                        help='train data name')

    #
    # Training parameters
    #
    parser.add_argument('--batch-size-33', type=int, default=128, help='batch size for training 33x33')
    parser.add_argument('--batch-size-99', type=int, default=64, help='batch size for training 99x99')
    parser.add_argument('--epochs', type=int, default=180, help='number of epochs')
    parser.add_argument('--finetune', type=int, default=50, help='number of epochs')

    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='constant',
                        choices=['constant', 'poly', 'step', 'design', 'design2'],
                        help='learning rate scheduler')
    parser.add_argument('--seed', type=int, default=43, help='seed')

    #
    # Save parameters
    #
    parser.add_argument('--val-per-epoch', type=int, default=1, help='interval of saving model')
    parser.add_argument('--log-per-iter', type=int, default=100, help='interval of logging')
    parser.add_argument('--base-dir', type=str, default='../result/', help='saving dir')

    #
    # Net parameters
    #
    parser.add_argument('--model-name', type=str, default='ctnet',
                        help='model name')
    parser.add_argument('--block-size', type=int, default=33,
                        help='block size')
    parser.add_argument('--cs-ratio', type=int, default=25,
                        choices=[1, 4, 10, 20, 25, 30, 40, 50], help='CS ratio')
    parser.add_argument('--stage-num', type=int, default=9,
                        help='stage number')
    parser.add_argument('--hem-layers', type=int, default=6,
                        help='hem layers')
    parser.add_argument('--hem-channel', type=int, default=32,
                        help='hem channel')


    args = parser.parse_args()

    args.total_epochs = args.epochs + args.finetune

    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime(time.time()))
    folder_name = '{}_{}_cs{}_stage{}_hemlayer{}_hemchannel{}'.format(
        args.time_name, args.model_name, args.cs_ratio, args.stage_num, args.hem_layers, args.hem_channel)
    args.save_folder = osp.join(args.base_dir, folder_name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # val
    args.val_dict = {
        'set11': osp.join(args.data_dir, 'Set11'),
        'bsd68': osp.join(args.data_dir, 'CBSD68'),
        'urban100': osp.join(args.data_dir, 'Urban100')
    }

    # logger
    args.logger = setup_logger("Compressive Sensing", args.save_folder, 0, filename='log.txt')
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0
        self.logger = args.logger
        self.ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

        ## dataset
        self.switch_dataset(patch_size=99)
        self.iter_per_epoch99 = len(self.train_data_loader)
        self.max_iter = self.args.finetune * self.iter_per_epoch99

        self.switch_dataset(patch_size=33)
        self.iter_per_epoch33 = len(self.train_data_loader)
        self.max_iter += self.args.epochs * self.iter_per_epoch33

        ## GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## model
        net_kwargs = {'stage_num': args.stage_num, 'n_input': self.ratio_dict[args.cs_ratio],
                      'hem_layers': args.hem_layers, 'hem_channel': args.hem_channel}
        self.net = get_model(args.model_name, **net_kwargs).to(self.device)
        # self.net.apply(self.weight_init)

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_data_loader), lr_step=40)

        ## optimizer
        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        ## evaluation metrics
        self.best_psnr = {}
        self.best_ssim = {}
        for data in args.val_dict:
            self.best_psnr[data] = 0
            self.best_ssim[data] = 0

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
                if suffix in ['.png', '.jpg', '.tif', '.tiff']:
                    val_img_paths[key].append(osp.join(data_path, filename))
        self.val_img_paths = val_img_paths

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=self.args.save_folder)

        ## log info
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

        ## test memory
        self.test_memory()

    def switch_dataset(self, patch_size=33):
        assert patch_size in [33, 99]
        data_name = self.args.data_name_33 if patch_size == 33 else self.args.data_name_99
        batch_size = self.args.batch_size_33 if patch_size == 33 else self.args.batch_size_99

        trainset = CsTrain(mat_path=osp.join(self.args.data_dir, data_name))
        self.train_data_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    def test_memory(self):
        self.net.eval()

        self.logger.info('...testing Memory: 33x33, batch size [{:d}].'.format(self.args.batch_size_33))
        test_data = torch.ones((self.args.batch_size_33, 1, 33, 33), dtype=torch.float)
        with torch.no_grad():
            _, _ = self.net(test_data.to(self.device))
        string = 'Passed: 33x33, batch size [{:d}].'.format(self.args.batch_size_33)
        if torch.cuda.is_available():
            string += ' Memory: {:.4f} GBs'.format(torch.cuda.max_memory_allocated() / 1024 ** 3)
        self.logger.info(string)

        self.logger.info('...testing Memory: 99x99, batch size [{:d}]'.format(self.args.batch_size_99))
        test_data = torch.ones((self.args.batch_size_99, 1, 99, 99), dtype=torch.float)
        with torch.no_grad():
            _, _ = self.net(test_data.to(self.device))
        string = 'Passed: 99x99, batch size [{:d}]'.format(self.args.batch_size_99)
        if torch.cuda.is_available():
            string += ' Memory: {:.4f} GBs'.format(torch.cuda.max_memory_allocated() / 1024 ** 3)
        self.logger.info(string)

    def training(self):
        # training step

        start_time = time.time()
        base_log = "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || Loss: {:.4f} " \
                   "|| Cost Time: {} || Estimated Time: {}"
        for epoch in range(self.args.total_epochs):
            # switch dataset for finetune
            if self.args.finetune > 0 and epoch == self.args.epochs:
                self.switch_dataset(patch_size=99)
            for i, data in enumerate(self.train_data_loader):
                self.net.train()
                self.scheduler(self.optimizer, i, epoch)

                data = data.to(self.device)
                pred, loss_phi = self.net(data)
                loss_discrepancy = torch.mean(torch.pow(pred - data, 2))

                gamma = torch.Tensor([0.01]).to(self.device)

                loss_all = loss_discrepancy + torch.mul(gamma, loss_phi)

                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Losses/All loss', loss_all, self.iter_num)
                self.writer.add_scalar('Losses/Dis loss', loss_discrepancy, self.iter_num)
                self.writer.add_scalar('Losses/Phi loss', loss_phi, self.iter_num)
                self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], self.iter_num)

                if epoch < self.args.epochs:
                    cur_iter_num = self.iter_num - epoch * self.iter_per_epoch33
                    cur_iter_per_epoch = self.iter_per_epoch33
                else:
                    cur_iter_num = self.iter_num - self.args.epochs * self.iter_per_epoch33 - \
                                   (epoch - self.args.epochs) * self.iter_per_epoch99
                    cur_iter_per_epoch = self.iter_per_epoch99

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(epoch+1, args.total_epochs, cur_iter_num, cur_iter_per_epoch,
                                        self.optimizer.param_groups[0]['lr'], loss_all.item(), cost_string, eta_string))

            if epoch % args.val_per_epoch == 0:
                self.validation()

    def validation(self):
        self.net.eval()
        base_log = "Data: {:s}, PSNR: {:.4f}/{:.4f}, SSIM: {:.4f}/{:.4f}"
        for data in self.val_img_paths:
            psnrs, ssims = [], []
            for path in self.val_img_paths[data]:
                rec_psnr, rec_ssim, rec_img = self.val_single_img(path)
                psnrs.append(rec_psnr)
                ssims.append(rec_ssim)

            self.logger.info(base_log.format(data, np.mean(psnrs), self.best_psnr[data],
                                             np.mean(ssims), self.best_ssim[data]))

            # save models
            latest_pkl_name = osp.join(args.save_folder, 'latest.pkl')
            torch.save(self.net.state_dict(), latest_pkl_name)
            if self.best_psnr[data] < np.mean(psnrs):
                self.best_psnr[data] = np.mean(psnrs)
                best_pkl_name = osp.join(args.save_folder, 'best_{:s}.pkl'.format(data))
                torch.save(self.net.state_dict(), best_pkl_name)
            if self.best_ssim[data] < np.mean(ssims):
                self.best_ssim[data] = np.mean(ssims)

            # write scalars
            self.writer.add_scalar('Val Real/{:s} PSNR'.format(data), np.mean(psnrs),
                                   self.iter_num // len(self.train_data_loader))
            self.writer.add_scalar('Val Real/{:s} SSIM'.format(data), np.mean(psnrs),
                                   self.iter_num // len(self.train_data_loader))
            self.writer.add_scalar('Val Best/{:s} PSNR'.format(data), self.best_psnr[data],
                                   self.iter_num // len(self.train_data_loader))
            self.writer.add_scalar('Val Best/{:s} SSIM'.format(data), self.best_ssim[data],
                                   self.iter_num // len(self.train_data_loader))

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
    trainer.training()





