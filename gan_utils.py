import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import pdb
import time
import os
import torch.nn.functional as F
from six.moves import cPickle
import torch.nn.init as init
from models.spectral_normalization import SpectralNorm
from torch.optim import lr_scheduler

try:
    from torch.nn.utils import spectral_norm
except:
    print("can not input spectral_norm")

def update_learning_rate(schedulers, optimizers, metric = None):
    """Update learning rates for all the networks; called at the end of every epoch"""
    for scheduler in schedulers:
        scheduler.step(metric)
    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    return lr

def get_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        # Define hidden linear layers
        self.use_spectral_norm = opt.use_spectral_norm
        if opt.gan_d_type == 0:
            self.map = nn.Sequential(spectral_norm(nn.Linear(opt.rnn_size, 1)) if self.use_spectral_norm else nn.Linear(opt.rnn_size, 1),
                                     nn.LeakyReLU(negative_slope=0.2))
        elif opt.gan_d_type == 1:
            self.map = nn.Sequential(spectral_norm(nn.Linear(opt.rnn_size, 64)) if self.use_spectral_norm else nn.Linear(opt.rnn_size, 64),
                                     nn.LeakyReLU(negative_slope=0.2))
        elif opt.gan_d_type == 2:
            self.map = nn.Sequential(spectral_norm(nn.Linear(opt.rnn_size, 128)) if self.use_spectral_norm else nn.Linear(opt.rnn_size, 128),
                                     nn.LeakyReLU(negative_slope=0.2))
        elif opt.gan_d_type == 3:
            self.map = nn.Sequential(spectral_norm(nn.Linear(opt.rnn_size, opt.rnn_size)) if self.use_spectral_norm else nn.Linear(opt.rnn_size, opt.rnn_size),
                                     nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = self.map(x)
        return x

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        if opt.gan_g_type == 0:
            self.map = nn.Sequential(nn.Linear(opt.rnn_size, opt.rnn_size, bias=False), nn.ReLU())
        elif opt.gan_g_type == 1:
            self.map = nn.Sequential(nn.Linear(opt.rnn_size, opt.rnn_size, bias=False), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = self.map(x)
        return x

##### Generator #####
def linear_block(in_features, out_features, batch_norm=False):
    layers = []
    layers.append(spectral_norm(nn.Linear(in_features, out_features)))
    return nn.Sequential(*layers)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cycle_GAN_backward_D(opt, fake_pool_obj, fake_pool_rel, fake_pool_atr, netD_obj, netD_rel, netD_atr, criterionGAN,  real_obj, real_rel, real_atr, fake_obj, fake_rel, fake_atr):
    """Calculate GAN loss for discriminator D_A"""
    loss_D = 0.0
    if netD_rel is not None:
        loss_D_rel = cycle_GAN_backward_D_single(opt, fake_pool_rel, netD_rel, criterionGAN, real_rel, fake_rel)
        loss_D = loss_D_rel + loss_D
    if netD_obj is not None:
        loss_D_obj = cycle_GAN_backward_D_single(opt, fake_pool_obj, netD_obj, criterionGAN, real_obj, fake_obj)
        loss_D = loss_D + loss_D_obj
    if netD_atr is not None:
        loss_D_atr = cycle_GAN_backward_D_single(opt, fake_pool_atr, netD_atr, criterionGAN, real_atr, fake_atr)
        loss_D = loss_D + loss_D_atr
    return loss_D

def cycle_GAN_backward_D_single(opt, fake_pool, netD_rel, criterionGAN, real_rel, fake_rel):
    """Calculate GAN loss for discriminator D_A"""
    loss_D_real = criterionGAN(netD_rel(real_rel), True)
    loss_D_fake = criterionGAN(netD_rel(fake_pool.query(fake_rel).detach()), False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    return loss_D


def paired_backward_G(opt, criterionGAN, criterionCycle, criterionIdt, domain_A, domain_B):
    """Calculate the loss for generators G_A and G_B"""
    [real_A_obj, real_A_rel, real_A_atr] = domain_A
    [real_B_obj, real_B_rel, real_B_atr] = domain_B

    loss_G = criterionIdt(real_A_obj, real_B_obj.detach())+ criterionIdt(real_A_rel, real_B_rel.detach())+ criterionIdt(real_A_atr, real_B_atr.detach())
    return loss_G

def cycle_GAN_backward_G(opt, criterionGAN, criterionCycle, criterionIdt,
                         netG_A_obj, netG_A_rel, netG_A_atr, netG_B_obj, netG_B_rel, netG_B_atr,
                         netD_A_obj, netD_A_rel, netD_A_atr, netD_B_obj, netD_B_rel, netD_B_atr, domain_A, domain_B):
    """Calculate the loss for generators G_A and G_B"""
    [real_A_obj, real_A_rel, real_A_atr,
     fake_A_obj, fake_A_rel, fake_A_atr,
     rec_A_obj, rec_A_rel, rec_A_atr,
     idt_A_obj, idt_A_rel, idt_A_atr] = domain_A
    [real_B_obj, real_B_rel, real_B_atr,
     fake_B_obj, fake_B_rel, fake_B_atr,
     rec_B_obj, rec_B_rel, rec_B_atr,
     idt_B_obj, idt_B_rel, idt_B_atr] = domain_B

    loss_G = 0.0
    # combined loss and calculate gradients
    if netD_A_rel is not None:
        loss_G_rel = cycle_GAN_backward_G_single(opt, criterionGAN, criterionCycle, criterionIdt, netD_A_rel, netD_B_rel, \
                                           [real_A_rel, fake_A_rel, rec_A_rel, idt_A_rel], [real_B_rel, fake_B_rel, rec_B_rel, idt_B_rel])
        loss_G = loss_G + loss_G_rel

    if netD_A_obj is not None:
        loss_G_obj = cycle_GAN_backward_G_single(opt, criterionGAN, criterionCycle, criterionIdt, netD_A_obj, netD_B_obj, \
                                                 [real_A_obj, fake_A_obj, rec_A_obj, idt_A_obj], [real_B_obj, fake_B_obj, rec_B_obj, idt_B_obj])
        loss_G = loss_G + loss_G_obj
    if netD_A_atr is not None:
        loss_G_atr = cycle_GAN_backward_G_single(opt, criterionGAN, criterionCycle, criterionIdt, netD_A_atr, netD_B_atr,\
                                            [real_A_atr, fake_A_atr, rec_A_atr, idt_A_atr], [real_B_atr, fake_B_atr, rec_B_atr, idt_B_atr])
        loss_G = loss_G + loss_G_atr

    if opt.use_orthogonal:
        oloss_a = 0.0
        if netG_A_rel is not None:
            oloss_a_rel = l2_reg_ortho(netG_A_rel)
            oloss_a = oloss_a + oloss_a_rel
        if netG_A_obj is not None:
            oloss_a_obj = l2_reg_ortho(netG_A_obj)
            oloss_a = oloss_a + oloss_a_obj
        if netG_A_atr is not None:
            oloss_a_atr = l2_reg_ortho(netG_A_rel)
            oloss_a = oloss_a + oloss_a_atr
        oloss_b = 0.0
        if netG_B_rel is not None:
            oloss_b_rel = l2_reg_ortho(netG_B_rel)
            oloss_b = oloss_b + oloss_b_rel
        if netG_B_obj is not None:
            oloss_b_obj = l2_reg_ortho(netG_B_obj)
            oloss_b = oloss_b + oloss_b_obj
        if netG_B_atr is not None:
            oloss_b_atr = l2_reg_ortho(netG_B_atr)
            oloss_b = oloss_b + oloss_b_atr

        loss_G = loss_G + oloss_a
        loss_G = loss_G + oloss_b

    return loss_G

def cycle_GAN_backward_G_single(opt, criterionGAN, criterionCycle, criterionIdt, netD_A, netD_B, domain_A, domain_B):
    """Calculate the loss for generators G_A and G_B"""
    [real_A, fake_A, rec_A, idt_A] = domain_A
    [real_B, fake_B, rec_B, idt_B] = domain_B

    # Identity loss
    if opt.lambda_idt > 0:
        loss_idt_A = criterionIdt(idt_A, real_B.detach()) * opt.lambda_B * opt.lambda_idt  # G_A should be identity if real_B is fed: ||G_A(B) - B||
        loss_idt_B = criterionIdt(idt_B, real_A.detach()) * opt.lambda_A * opt.lambda_idt  # G_B should be identity if real_A is fed: ||G_B(A) - A||
    else:
        loss_idt_A = 0
        loss_idt_B = 0

    loss_G_A = criterionGAN(netD_A(fake_B), True)    # GAN loss D_A(G_A(A))
    loss_G_B = criterionGAN(netD_B(fake_A), True)     # GAN loss D_B(G_B(B))
    loss_cycle_A = criterionCycle(rec_A, real_A.detach()) * opt.lambda_A  # Forward cycle loss || G_B(G_A(A)) - A||
    loss_cycle_B = criterionCycle(rec_B, real_B.detach()) * opt.lambda_B  # Backward cycle loss || G_A(G_B(B)) - B||

    # combined loss and calculate gradients
    loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
    return loss_G

def kaiming_normal_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

def GAN_init_G(opt, net, type=None):
    if vars(opt).get('start_from_gan', None) is not None:
        assert os.path.isdir(opt.checkpoint_path_gan)," %s must be a a path" % opt.start_from_gan
        try:
            if type is not None:
                net.load_state_dict(torch.load(os.path.join(opt.checkpoint_path_gan, 'model_G-best.pth'))[type])
            else:
                print("kaiming_normal_init:" + type)
                net = kaiming_normal_init(net)
            print('Init from {}'.format(os.path.join(opt.checkpoint_path_gan, 'model_G-best.pth')))
        except:
            print("Can not load checkpoint")
            print("kaiming_normal_init:" + type)
            net = kaiming_normal_init(net)
    else:
        if vars(opt).get('init_from', None) is not None:
            try:
                net.load_state_dict(torch.load(os.path.join(opt.init_from, 'model_G-best.pth'))[type])
                print('Init from {}'.format(os.path.join(opt.init_from, 'model_G-best.pth')))
            except:
                print("kaiming_normal_init:" + type)
                net = kaiming_normal_init(net)
        else:
            print("kaiming_normal_init:" + type)
            net = kaiming_normal_init(net)
    return net.float()

def GAN_init_D(opt, net, type=None):
    if vars(opt).get('start_from_gan', None) is not None:
        assert os.path.isdir(opt.checkpoint_path_gan)," %s must be a a path" % opt.start_from_gan
        try:
            if type is not None:
                net.load_state_dict(torch.load(os.path.join(opt.checkpoint_path_gan, 'model_D-best.pth'))[type])
            else:
                print("kaiming_normal_init:" + type)
                net = kaiming_normal_init(net)
            print('Load from {}'.format(os.path.join(opt.checkpoint_path_gan, 'model_D-best.pth')))
        except:
            print("Can not load checkpoint")
            print("kaiming_normal_init:" + type)
            net = kaiming_normal_init(net)
    else:
        if vars(opt).get('init_from', None) is not None:
            try:
                net.load_state_dict(torch.load(os.path.join(opt.init_from, 'model_D-best.pth'))[type])
                print('Init from {}'.format(os.path.join(opt.init_from, 'model_D-best.pth')))
            except:
                print("kaiming_normal_init:" + type)
                net = kaiming_normal_init(net)
        else:
            print("kaiming_normal_init:" + type)
            net = kaiming_normal_init(net)
    return net.float()

"""Function used for Orthogonal Regularization"""
# https://github.com/nbansal90/Can-we-Gain-More-from-Orthogonality/blob/master/Wide-Resnet/train_n.py
def l2_reg_ortho(mdl):
        l2_reg = None
        for W in mdl.parameters():
                if W.ndimension() < 2:
                        continue
                else:
                        cols = W[0].numel()
                        rows = W.shape[0]
                        w1 = W.view(-1,cols)
                        wt = torch.transpose(w1,0,1)
                        if (rows > cols):
                                m  = torch.matmul(wt,w1)
                                ident = torch.eye(cols,cols)
                        else:
                                m = torch.matmul(w1,wt)
                                ident = torch.eye(rows,rows)

                        ident = ident.cuda()
                        w_tmp = (m - ident)
                        b_k = torch.rand(w_tmp.shape[1],1)
                        b_k = b_k.cuda()

                        v1 = torch.matmul(w_tmp, b_k)
                        norm1 = torch.norm(v1,2)
                        v2 = torch.div(v1,norm1)
                        v3 = torch.matmul(w_tmp,v2)

                        if l2_reg is None:
                                l2_reg = (torch.norm(v3,2))**2
                        else:
                                l2_reg = l2_reg + (torch.norm(v3,2))**2
        return l2_reg