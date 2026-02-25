# coding:utf-8
import cv2
import torch
import numpy as np
from networks.ResUnet import ResUnet
from utils.metrics import calculate_metrics
import imageio
import os
import pandas as pd
from utils.metrics import AverageMeter
from torchmetrics import F1, Accuracy, AUROC, Specificity


class Test:
    def __init__(self, config, test_loader):
        # 数据加载
        self.test_loader = test_loader

        # 模型
        self.model = None
        self.model_type = config.model_type

        # 路径设置
        self.target = config.Target_Dataset
        self.result_path = config.result_path
        self.model_path = config.model_path

        # 其他
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.mode = config.mode
        self.device = config.device

        self.n_domains = config.n_domains
        self.domain_discriminator_flag = config.domain_discriminator_flag
        self.grl = config.grl
        self.lambd = config.lambd
        self.drop_percent = config.drop_percent
        self.filter_WRS_flag = config.filter_WRS_flag
        self.recover_flag = config.recover_flag

        self.domain_loss_flag = config.domain_loss_flag
        self.discriminator_layers = config.discriminator_layers

        self.layer_wise_prob = config.layer_wise_prob

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet='resnet34', num_classes=self.out_ch, pretrained=True,
                                 n_domains=self.n_domains, domain_discriminator_flag=0, grl=self.grl, lambd=self.lambd,
                                 drop_percent=self.drop_percent, filter_WRS_flag=self.filter_WRS_flag,
                                 recover_flag=self.recover_flag
                                 ).to(self.device)
        else:
            raise ValueError('The model type is wrong!')

        checkpoint = torch.load(self.model_path + '/' + 'now' + '-' + self.model_type + '.pth',
                                map_location=lambda storage, loc: storage.cuda(0))
        self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def test(self):
        print("Testing and Saving the results...")
        print("--" * 15)

        val_dsc = AverageMeter()
        val_acc = AverageMeter()
        val_sp = AverageMeter()
        val_se = AverageMeter()
        val_aucroc = AverageMeter()

        f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')
        auroc = AUROC().cuda()
        accuracy = Accuracy().cuda()
        sp_se = Specificity().cuda()

        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                x, y, roi = data['image'], data['label'], data['roi']
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                roi = roi.cuda(non_blocking=True)



                seg_output, domain_logit, self_distillation_loss = self.model(x)
                seg_soft = torch.sigmoid(seg_output)
                seg_hard = seg_soft.clone().detach()
                masked_seg_soft = torch.masked_select(seg_soft, roi)
                masked_seg_hard = torch.masked_select(seg_hard, roi)
                masked_mask_gt = torch.masked_select(y, roi)
                dsc = f1_score(torch.stack([1 - seg_hard[:, 0], seg_hard[:, 0]], dim=1), y[:, 0].long())[1]
                acc = accuracy(torch.stack([1 - masked_seg_hard, masked_seg_hard], dim=0).unsqueeze(0),
                               masked_mask_gt.unsqueeze(0).long())
                aucroc = auroc(masked_seg_soft, masked_mask_gt.long())
                sp = sp_se(masked_seg_hard, masked_mask_gt.long())
                se = sp_se(1 - masked_seg_hard, (1 - masked_mask_gt).long())

                val_dsc.update(dsc.item(), x.size(0))
                val_acc.update(acc.item(), x.size(0))
                val_aucroc.update(aucroc.item(), x.size(0))
                val_sp.update(sp.item(), x.size(0))
                val_se.update(se.item(), x.size(0))

        return val_dsc.avg, val_acc.avg, val_aucroc.avg, val_sp.avg, val_se.avg
