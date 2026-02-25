from torchnet import meter
from networks.ResUnet import ResUnet
from config import *
import numpy as np
from tensorboardX import SummaryWriter
from test import Test
import datetime
import random



class TrainDG:
    def __init__(self, config, train_loader, valid_loader=None):
        # 数据加载
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 模型
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type

        # 损失函数
        self.seg_cost = Seg_loss()

        # 优化器
        self.optimizer = None
        self.scheduler = None
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # 训练设置
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # 路径设置
        self.model_path = config.model_path
        self.result_path = config.result_path

        # 其他
        self.log_path = config.log_path
        self.warm_up = -1
        self.valid_frequency = 1   # 多少轮测试一次
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
        self.domain_criterion = nn.CrossEntropyLoss()

        self.build_model()
        self.print_network()

    def build_model(self):
        if self.model_type == 'Res_Unet':
            self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=True,
                                 n_domains=self.n_domains, domain_discriminator_flag=self.domain_discriminator_flag,
                                 grl=self.grl, lambd=self.lambd,
                                 drop_percent=self.drop_percent, filter_WRS_flag=self.filter_WRS_flag,
                                 recover_flag=self.recover_flag
                                 ).to(self.device)
        else:
            raise ValueError('The model type is wrong!')

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        if torch.cuda.device_count() > 1:
            device_ids = list(range(0, torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        if self.lr_scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
        elif self.lr_scheduler == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        elif self.lr_scheduler == 'Epoch':
            self.scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
        else:
            self.scheduler = None

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def select_layers(self, layer_wise_prob):
        layer_index = np.random.randint(len(self.discriminator_layers), size=1)[0]
        layer_select = self.discriminator_layers[layer_index]
        layer_drop_flag = [0, 0, 0, 0, 0]
        if random.random() <= layer_wise_prob:
            layer_drop_flag[layer_select] = 1
        return layer_drop_flag

    def run(self):
        writer = SummaryWriter(self.log_path.replace('.log', '.writer'))
        print("Training...")
        best_loss, best_epoch = np.inf, 0
        loss_meter = meter.AverageValueMeter()
        best_dice = 0

        CE_domain_loss = [0.0 for i in range(5)]
        global_step = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()

            start_time = datetime.datetime.now()

            for batch, data in enumerate(self.train_loader):
                x, y = data['image'], data['label']
                domain_label = data['dc']
                domain_label = torch.tensor(domain_label, dtype=torch.long)

                x, y = x.to(self.device), y.to(self.device)
                domain_l = domain_label.to(self.device)

                layer_drop_flag = self.select_layers(layer_wise_prob=self.layer_wise_prob)

                pred, domain_logit, self_distillation_loss = self.model(input=x, domain_labels=domain_l, layer_drop_flag=layer_drop_flag, step=global_step)
                global_step += 1

                seg_loss = self.seg_cost(pred, y)

                domain_losses_avg = torch.tensor(0.0).to(device=self.device)
                if self.domain_discriminator_flag == 1:
                    domain_losses = []
                    for i, logit in enumerate(domain_logit):
                        domain_loss = self.domain_criterion(logit, domain_l)
                        domain_losses.append(domain_loss)
                        CE_domain_loss[i] += domain_loss
                    domain_losses = torch.stack(domain_losses, dim=0)
                    domain_losses_avg = domain_losses.mean(dim=0)

                if self.domain_loss_flag == 1:
                    loss = seg_loss + domain_losses_avg + 0.1 * self_distillation_loss
                else:
                    loss = seg_loss + 0.1 * self_distillation_loss

                self.optimizer.zero_grad()
                loss.backward()
                loss_meter.add(loss.sum().item())
                self.optimizer.step()



            if self.scheduler is not None:
                self.scheduler.step()

            print("Train ———— Total_Loss:{:.8f}".format(loss_meter.value()[0]))
            writer.add_scalar('total_loss_epoch', loss_meter.value()[0], epoch + 1)

            torch.save(self.model.state_dict(), self.model_path + '/' + 'now' + '-' + self.model_type + '.pth')

            if (epoch + 1) % self.valid_frequency == 0 and self.valid_loader is not None:
                test = Test(config=self.config, test_loader=self.valid_loader)
                dsc, acc, aucroc, sp, se = test.test()
                print("Dice:{}, acc:{}, aucroc:{}, sp:{}, se:{}".format(dsc, acc, aucroc, sp, se))
                if dsc > best_dice:
                    best_dice = dsc
                    best_epoch = (epoch + 1)
                    torch.save(self.model.state_dict(),
                               self.model_path + '/' + 'best' + '-' + self.model_type + '.pth')

                writer.add_scalar('Valid Dice', dsc, (epoch + 1) // self.valid_frequency)
                writer.add_scalar('Valid ACC', acc, (epoch + 1) // self.valid_frequency)
                writer.add_scalar('Valid aucroc', aucroc, (epoch + 1) // self.valid_frequency)
                writer.add_scalar('Valid sp', sp, (epoch + 1) // self.valid_frequency)
                writer.add_scalar('Valid se', se, (epoch + 1) // self.valid_frequency)

            print("best_dice:{}".format(best_dice))
            end_time = datetime.datetime.now()
            time_cost = end_time - start_time
            print('This epoch took {:6f} s'.format(time_cost.seconds + time_cost.microseconds / 1000000.))
            print("===" * 10)


        torch.save(self.model.state_dict(), self.model_path + '/' + 'last' + '-' + self.model_type + '.pth')
        print('The best dice:{} epoch:{}'.format(best_dice, best_epoch))
        print('global_step:{}'.format(global_step))
        writer.close()





