import argparse
import os
import random

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from tqdm import tqdm

from model import ResnetGenerator, StainPresetNet, NLayerDiscriminator, GANloss, init_weights, lr_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def print_options(opt, mparser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = mparser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def train(opt):
    dataset = ImageFolder(opt.train_dir_root,
                          transform=Compose([Resize(size=opt.train_size),
                                             ToTensor(),
                                             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
    print('len(dataset.classes):', len(dataset.classes))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize,
        shuffle=True, num_workers=opt.nThreads,
        drop_last=True)
    net_G_A = StainPresetNet(backbone=opt.backbone, resample_size=opt.resample_size,
                             channels=opt.channels, layers=opt.n_layer).to(device)
    net_T_list = []
    net_D_list = []
    for i in range(len(dataset.classes)):
        net_D_list.append(NLayerDiscriminator(3, 32, 3, norm_layer=nn.InstanceNorm2d).to(device))
        net_T_list.append(ResnetGenerator(3, 3, 64, norm_layer=nn.InstanceNorm2d, n_blocks=6).to(device))
    for i in range(len(net_D_list)):
        init_weights(net_D_list[i], opt.init_type, opt.init_gain)
        init_weights(net_T_list[i], opt.init_type, opt.init_gain)

    init_weights(net_G_A, opt.init_type, opt.init_gain)

    loss_fn = GANloss()
    loss_l1 = torch.nn.L1Loss()
    loss_ssim = kornia.losses.SSIMLoss(window_size=11)

    optim_G = optim.Adam(net_G_A.parameters(),
                         lr=opt.lr_G,
                         betas=opt.betas)
    optim_D_list = []
    optim_T_list = []
    for i in range(len(net_D_list)):
        optim_D_list.append(
            optim.Adam(net_D_list[i].parameters(),
                       lr=opt.lr_D,
                       betas=opt.betas)
        )
        optim_T_list.append(
            optim.Adam(net_T_list[i].parameters(),
                       lr=opt.lr_G,
                       betas=opt.betas)
        )
    step_num = opt.total_steps - opt.warmup_step
    sched_G = optim.lr_scheduler.CosineAnnealingLR(optim_G, T_max=step_num)
    sched_D_list = []
    sched_T_list = []
    for i in range(len(optim_D_list)):
        sched_D_list.append(optim.lr_scheduler.CosineAnnealingLR(optim_D_list[i], T_max=step_num))
        sched_T_list.append(optim.lr_scheduler.CosineAnnealingLR(optim_T_list[i], T_max=step_num))
    looper = infiniteloop(dataloader)
    # vis = Visualizer(opt.name)
    step_now = 0
    fs = open(os.path.join(opt.checkpoints_dir, opt.name, opt.name + '_log.txt'), 'a')
    if opt.pretrained is not None:
        static_param = torch.load(opt.pretrained)['net_G_A']
        net_G_A.load_state_dict(static_param)
        print(opt.pretrained, 'loaded')

    step = 0
    pbar = tqdm(total=opt.total_steps)
    while True:
        net_G_A.train()
        real_a, domain_id_a = next(looper)
        real_b, domain_id_b = next(looper)
        real_a = (real_a.to(device) - 0.5) * 2
        real_b = (real_b.to(device) - 0.5) * 2
        imsize_now = real_a.size(3)
        domain_id_a = domain_id_a[0]
        domain_id_b = domain_id_b[0]
        if domain_id_a == domain_id_b:
            continue
        step += 1
        pbar.update(1)
        if step > opt.total_steps:
            break
        if opt.random_scale == 1:
            scale = random.randint(8, imsize_now // 8) * 8
            real_a = torch.nn.functional.interpolate(real_a, size=(scale, scale), mode='bilinear',
                                                     align_corners=True)
            real_b = torch.nn.functional.interpolate(real_b, size=(scale, scale), mode='bilinear',
                                                     align_corners=True)

        set_requires_grad(net_D_list, False)
        # Generator
        fake_b = net_G_A(real_a, real_b)
        fake_bb = net_T_list[domain_id_b](fake_b)
        rec_aa = net_G_A(fake_bb, real_a)
        loss_gan = loss_fn(net_D_list[domain_id_b](fake_bb))
        loss_cycle = loss_l1(rec_aa, real_a) * opt.lambda_A
        loss_G = loss_gan + loss_cycle
        if opt.lambda_diff > 0:
            loss_diff = loss_l1(fake_b, fake_bb.detach()) * opt.lambda_diff
            loss_G += loss_diff
        if opt.lambda_structural > 0:
            loss_dom_struct = loss_ssim(fake_b, fake_bb.detach()) * opt.lambda_structural
            loss_G += loss_dom_struct
        if opt.lambda_identity > 0:
            loss_idt = 0
            idt_b = net_G_A(real_b, real_b)
            idt_bb = net_T_list[domain_id_b](real_b)
            loss_idt += loss_l1(idt_bb, real_b) * opt.lambda_identity
            loss_idt += loss_l1(idt_b, real_b) * opt.lambda_identity
            loss_G += loss_idt

        optim_G.zero_grad()
        optim_T_list[domain_id_b].zero_grad()
        loss_G.backward()
        optim_G.step()
        optim_T_list[domain_id_b].step()

        set_requires_grad(net_D_list, True)
        optim_D_list[domain_id_b].zero_grad()
        net_D_A_real = net_D_list[domain_id_b](real_b)
        net_D_A_fake = net_D_list[domain_id_b](fake_bb.detach())
        loss_D = loss_fn(net_D_A_real, net_D_A_fake)
        loss_D.backward()
        optim_D_list[domain_id_b].step()
        if step % opt.display_freq == 0:
            loss_dict = {
                'loss_GAN': float(loss_gan),
                'loss_D': float(loss_D),
                'loss_cycle': float(loss_cycle),
            }
            if opt.lambda_identity > 0:
                loss_dict['loss_idt'] = float(loss_idt)
                # vis.img('idt_b', idt_b[0] * 0.5 + 0.5)
                # vis.img('idt_bb', idt_bb[0] * 0.5 + 0.5)
            if opt.lambda_diff > 0:
                loss_dict['loss_diff'] = float(loss_diff)
            if opt.lambda_structural > 0:
                loss_dict['loss_dom_struct'] = float(loss_dom_struct)
            # vis.plot_many_in_one('L1 loss', loss_dict_l1)
            fs.write('step {},loss: {}\n'.format(step, loss_dict))
            fs.flush()
        if step_now < opt.warmup_step:
            lr_warmup(optim_G, step_now, opt.warmup_step, 0, opt.lr_G)
            for i in range(len(optim_D_list)):
                lr_warmup(optim_D_list[i], step_now, opt.warmup_step, 0, opt.lr_D)
                lr_warmup(optim_T_list[i], step_now, opt.warmup_step, 0, opt.lr_G)
            step_now += 1
        else:
            sched_G.step()
            for i in range(len(optim_D_list)):
                sched_T_list[i].step()
                sched_D_list[i].step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="StainPresetNet", type=str,
                        help="name of the experiment.")
    parser.add_argument("--train_dir_root",
                        default="/home/khtao/Dataset/StainPreset_Dataset/all_histopathology",
                        type=str,
                        help="path to images dir root for training")
    parser.add_argument("--train_size", default=256, type=int, help="image size during training")
    parser.add_argument("--test_size", default=256, type=int, help="image size during testing")
    parser.add_argument("--total_steps", default=2000000, type=int, help="total number of training steps")
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument("--lr_G", default=1e-4, type=float, help="Generator learning rate")
    parser.add_argument("--lr_D", default=1e-4, type=float, help="Discriminator learning rate")
    parser.add_argument("--betas", default=[0.5, 0.999], type=list, help="for Adam")
    parser.add_argument("--random_scale", default=1, type=int, help="use random sacle train Discriminator")
    parser.add_argument('--backbone', type=str, default='resnet18', help='the backbone of StainPresetNet')
    parser.add_argument('--resample_size', type=int, default=128, help='# of resample_size in StainPresetNet')
    parser.add_argument('--channels', type=int, default=8, help='# of channels in StainPresetNet')
    parser.add_argument('--pretrained', type=str,
                        default=None,
                        help='load pretrained StainPresetNet')
    parser.add_argument('--step_count', type=int, default=1, help='step count')
    parser.add_argument('--n_layer', type=int, default=2, help='# of layers in StainPresetNet')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.002,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--warmup_step', type=int, default=1000, help='learning rate warmup step')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_diff', type=float, default=10.0,
                        help='weight for diff StainPresetNet and resnetgereator')
    parser.add_argument('--lambda_identity', type=float, default=2.0,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                             ' For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss,'
                             ' please set lambda_identity = 0.1')
    parser.add_argument('--lambda_structural', type=float, default=1.0, help='weight for domain ssim')
    parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--device', type=str, default='1', help='run on # GPU')
    args = parser.parse_args()
    print_options(args, parser)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0')
    set_seed(args.seed)
    train(opt=args)
