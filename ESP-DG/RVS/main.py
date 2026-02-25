import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys, traceback
import datetime
import random
import numpy as np
import torch
import argparse

from train_DG import TrainDG
from test import Test
from dataloaders.RVS_dataloader import get_seg_dg_dataloader



torch.set_num_threads(1)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')
        self.hook = sys.excepthook
        sys.excepthook = self.kill

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def kill(self, ttype, tvalue, ttraceback):
        for trace in traceback.format_exception(ttype, tvalue, ttraceback):
            print(trace)
        os.remove(self.filename)

    def flush(self):
        pass


def print_information(config):
    print('GPUs: ' + str(torch.cuda.device_count()))
    print('time: ' + str(config.time_now))
    print('mode: ' + str(config.mode))
    print('source domain: ' + str(config.Source_Dataset))
    print('target domain: ' + str(config.Target_Dataset))
    print('model: ' + str(config.model_type))

    print('input size: ' + str(config.image_size))
    print('batch size: ' + str(config.batch_size))

    print('optimizer: ' + str(config.optimizer))
    print('lr_scheduler: ' + str(config.lr_scheduler))
    print('lr: ' + str(config.lr))
    print('momentum: ' + str(config.momentum))
    print('weight_decay: ' + str(config.weight_decay))
    print('***' * 10)


def main(config):
    seed_torch(0)
    config.time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")

    if config.load_time is not None:
        config.model_path = os.path.join(config.path_save_model, config.load_time)
    else:
        config.model_path = os.path.join(config.path_save_model, config.time_now)

    config.result_path = os.path.join(config.path_save_result, config.time_now, config.mode)
    config.log_path = os.path.join(config.path_save_log, config.mode)
    config.savefig = config.model_type+config.time_now

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    config.log_path = os.path.join(config.log_path, config.time_now+'.log')
    sys.stdout = Logger(config.log_path, sys.stdout)


    if config.mode == 'train_DG':
        print('Training Phase')

        print_information(config)

        source_dataloader, valid_loader = get_seg_dg_dataloader(config.dataset_root, config.Source_Dataset, config.Target_Dataset, config.batch_size, config.num_workers)
        train_DG = TrainDG(config, source_dataloader, valid_loader)
        train_DG.run()

    elif config.mode == 'single_test':
        print(config.Target_Dataset)
        print('Loading model: ' + str(config.load_time) + '/' + 'best' + '-' + str(config.model_type) + '.pth')

        source_dataloader, test_dataloader = get_seg_dg_dataloader(config.dataset_root, config.Source_Dataset, config.Target_Dataset, config.batch_size, config.num_workers)

        test = Test(config, test_dataloader)
        test.test()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_DG',
                        help='train_DG/train_kfold/single_test/multi_test')   # choose the mode of train_DG/test

    parser.add_argument('--kfold', type=int, default=3)
    parser.add_argument('--load_time', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='Res_Unet', help='Res_Unet')  # choose the model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')


    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam/AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='Epoch',
                        help='Cosine/Step/Epoch')   # choose the decrease strategy of lr
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # weight_decay in SGD
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam/AdamW
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam/AdamW
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=14)

    parser.add_argument("--layer_wise_prob", default=1, type=float, help="prob to use layer-wise dropout")
    parser.add_argument('--n_domains', type=int, default=3, help="numbers of source domains.")  ##############
    parser.add_argument("--domain_discriminator_flag", default=1, type=int, help="whether use domain discriminator.")
    parser.add_argument("--domain_loss_flag", default=1, type=int, help="whether use domain loss.")
    parser.add_argument("--discriminator_layers", default=[0, 1, 2, 3, 4], nargs="+", type=int,
                        help="where to place discriminators")
    parser.add_argument("--grl", default=1, type=int, help="whether to use grl")
    parser.add_argument("--lambd", default=0.25, type=float, help="weight of grl")

    parser.add_argument("--drop_percent", default=0.2, type=float, help="percent of dropped filters")
    parser.add_argument("--filter_WRS_flag", default=1, type=int, help="Weighted Random Selection.")
    parser.add_argument("--recover_flag", default=1, type=int)

    parser.add_argument('--Source_Dataset', nargs='+', type=str, default=[])
    parser.add_argument('--Target_Dataset', type=str, default=[])

    parser.add_argument('--path_save_result', type=str, default='')
    parser.add_argument('--path_save_model', type=str, default='')
    parser.add_argument('--path_save_log', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='')

    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda')
    else:
        parser.add_argument('--device', type=str, default='cpu')

    config = parser.parse_args()
    main(config)


