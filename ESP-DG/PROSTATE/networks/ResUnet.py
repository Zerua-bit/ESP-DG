from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.unet import UnetBlock


class KLDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super().__init__()
        self.T = temperature
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits):
        log_p = F.log_softmax(student_logits / self.T, dim=1)
        q = F.softmax(teacher_logits / self.T, dim=1)
        loss = F.kl_div(log_p, q, reduction=self.reduction) * (self.T ** 2)
        return loss




class DownUpAlign(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, scale_factor=2):
        super().__init__()
        self.down = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.out_proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = self.out_proj(x)
        return x




class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False, n_domains=5, domain_discriminator_flag=1, grl=1, lambd=0.25, drop_percent=0.33, filter_WRS_flag=1, recover_flag=1):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.res = base_model(pretrained=pretrained, domains=n_domains,
                              domain_discriminator_flag=domain_discriminator_flag,
                              grl=grl,
                              lambd=lambd,
                              drop_percent=drop_percent,
                              wrs_flag=filter_WRS_flag,
                              recover_flag=recover_flag, )

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

        self.DownUpAlign_3 = DownUpAlign(in_channels=128, mid_channels=32, out_channels=self.num_classes, scale_factor=16)
        self.DownUpAlign_2 = DownUpAlign(in_channels=128, mid_channels=32, out_channels=self.num_classes, scale_factor=8)
        self.DownUpAlign_1 = DownUpAlign(in_channels=128, mid_channels=32, out_channels=self.num_classes, scale_factor=4)
        self.DownUpAlign_0 = DownUpAlign(in_channels=128, mid_channels=32, out_channels=self.num_classes, scale_factor=2)
        self.self_distillation_loss = KLDistillationLoss()



    def forward(self, input, domain_labels=None, layer_drop_flag=None, step=0):
        sfs, domain_logit = self.res(x=input, domain_labels=domain_labels, layer_drop_flag=layer_drop_flag, step=step)
        sfs[4] = F.relu(sfs[4])

        sfs[4], y_3 = self.up1(sfs[4], sfs[3])
        sfs[4], y_2 = self.up2(sfs[4], sfs[2])
        sfs[4], y_1 = self.up3(sfs[4], sfs[1])
        sfs[4], y_0 = self.up4(sfs[4], sfs[0])
        sfs[4] = self.up5(sfs[4])
        head_input = F.relu(self.bnout(sfs[4]))
        seg_output = self.seg_head(head_input)

        y_3 = self.DownUpAlign_3(y_3)
        y_2 = self.DownUpAlign_2(y_2)
        y_1 = self.DownUpAlign_1(y_1)
        y_0 = self.DownUpAlign_0(y_0)

        self_distillation_loss_3 = self.self_distillation_loss(y_3, seg_output)
        self_distillation_loss_2 = self.self_distillation_loss(y_2, seg_output)
        self_distillation_loss_1 = self.self_distillation_loss(y_1, seg_output)
        self_distillation_loss_0 = self.self_distillation_loss(y_0, seg_output)

        self_distillation_loss = (self_distillation_loss_3 + self_distillation_loss_2 + self_distillation_loss_1 + self_distillation_loss_0) / 4

        return seg_output, domain_logit, self_distillation_loss

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=1, pretrained=False, mixstyle_layers=['layer1'], random_type='Random')
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 1, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

