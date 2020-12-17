from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import datetime
import yaml
import io

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
import opts
import models
from dataloader import *
from dataloader_up_mt import *
from dataloader_gan import *
import eval_utils_gan
import misc.utils as utils
from misc.rewards_up import init_scorer, get_self_critical_reward
from gan_utils import *

try:
    import tensorflow as tf
except ImportError:
    pdb.set_trace()
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

lambda_obj = lambda_rel = lambda_atr = 1.0

def add_summary_value(writer, keys, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=keys, simple_value=value)])
    writer.add_summary(summary, iteration)


def train(opt):
    if vars(opt).get('start_from', None) is not None:
        opt.checkpoint_path = opt.start_from
        opt.id = opt.checkpoint_path.split('/')[-1]
        print('Point to folder: {}'.format(opt.checkpoint_path))
    else:
        opt.id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + opt.caption_model
        opt.checkpoint_path = os.path.join(opt.checkpoint_path, opt.id)

        if not os.path.exists(opt.checkpoint_path): os.makedirs(opt.checkpoint_path)
        print('Create folder: {}'.format(opt.checkpoint_path))

    # Write YAML file
    with io.open(opt.checkpoint_path + '/opts.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(opt, outfile, default_flow_style=False, allow_unicode=True)

    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = DataLoader_GAN(opt)
    loader_i2t = DataLoader_UP(opt)

    opt.vocab_size = loader.vocab_size
    if opt.use_rela == 1:
        opt.rela_dict_size = loader.rela_dict_size
    opt.seq_length = loader.seq_length
    use_rela = getattr(opt, 'use_rela', 0)

    try:
        tb_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)
    except:
        print('Set tensorboard error!')
        pdb.set_trace()

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        try:
            with open(os.path.join(opt.checkpoint_path, 'infos.pkl')) as f:
                infos = cPickle.load(f)
                saved_model_opt = infos['opt']
                need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
                for checkme in need_be_same:
                    assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

            if os.path.isfile(os.path.join(opt.checkpoint_path, 'histories.pkl')):
                with open(os.path.join(opt.checkpoint_path, 'histories.pkl')) as f:
                    histories = cPickle.load(f)
        except:
            print("Can not load infos.pkl")

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # opt.caption_model = 'up_gtssg_sep_self_att_sep'
    opt.caption_model = opt.caption_model_to_replace
    model = models.setup(opt).cuda()
    print('### Model summary below###\n {}\n'.format(str(model)))
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameter:{}'.format(model_params))

    model.eval()

    train_loss = 0
    update_lr_flag = True
    fake_A_pool_obj = utils.ImagePool(opt.pool_size)
    fake_A_pool_rel = utils.ImagePool(opt.pool_size)
    fake_A_pool_atr = utils.ImagePool(opt.pool_size)

    fake_B_pool_obj = utils.ImagePool(opt.pool_size)
    fake_B_pool_rel = utils.ImagePool(opt.pool_size)
    fake_B_pool_atr = utils.ImagePool(opt.pool_size)

    netD_A_obj = GAN_init_D(opt, Discriminator(opt), type='netD_A_obj').cuda().train()
    netD_A_rel = GAN_init_D(opt, Discriminator(opt), type='netD_A_rel').cuda().train()
    netD_A_atr = GAN_init_D(opt, Discriminator(opt), type='netD_A_atr').cuda().train()

    netD_B_obj = GAN_init_D(opt, Discriminator(opt), type='netD_B_obj').cuda().train()
    netD_B_rel = GAN_init_D(opt, Discriminator(opt), type='netD_B_rel').cuda().train()
    netD_B_atr = GAN_init_D(opt, Discriminator(opt), type='netD_B_atr').cuda().train()

    netG_A_obj = GAN_init_G(opt, Generator(opt), type='netG_A_obj').cuda().train()
    netG_A_rel = GAN_init_G(opt, Generator(opt), type='netG_A_rel').cuda().train()
    netG_A_atr = GAN_init_G(opt, Generator(opt), type='netG_A_atr').cuda().train()

    netG_B_obj = GAN_init_G(opt, Generator(opt), type='netG_B_obj').cuda().train()
    netG_B_rel = GAN_init_G(opt, Generator(opt), type='netG_B_rel').cuda().train()
    netG_B_atr = GAN_init_G(opt, Generator(opt), type='netG_B_atr').cuda().train()

    optimizer_G = utils.build_optimizer(itertools.chain(netG_A_obj.parameters(), netG_B_obj.parameters(),
                                                        netG_A_rel.parameters(), netG_B_rel.parameters(),
                                                        netG_A_atr.parameters(), netG_B_atr.parameters()), opt)
    optimizer_D = utils.build_optimizer(itertools.chain(netD_A_obj.parameters(), netD_B_obj.parameters(),
                                                        netD_A_rel.parameters(), netD_B_rel.parameters(),
                                                        netD_A_atr.parameters(), netD_B_atr.parameters()), opt)

    criterionGAN = GANLoss(opt.gan_mode).cuda()  # define GAN loss.
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    optimizers = []
    optimizers.append(optimizer_G)
    optimizers.append(optimizer_D)
    schedulers = [get_scheduler(opt, optimizer) for optimizer in optimizers]
    current_lr = optimizers[0].param_groups[0]['lr']
    train_num = 0
    update_lr_flag = True

    while True:
        if update_lr_flag and opt.current_lr>= 1e-4:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            if opt.current_lr>= 1e-4:
                utils.set_lr(optimizer, opt.current_lr)
            else:
                utils.set_lr(optimizer, 1e-4)
            update_lr_flag = False

        """
        Show the percentage of data loader
        """
        if train_num > loader.max_index:
            train_num = 0
        train_num = train_num + 1
        train_precentage = float(train_num)*100/float(loader.max_index)
        """
        Start training
        """
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)
        # print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['isg_feats'][:, 0, :], data['isg_feats'][:, 1, :], data['isg_feats'][:, 2, :],
               data['ssg_feats'][:, 0, :], data['ssg_feats'][:, 1, :], data['ssg_feats'][:, 2, :]]

        tmp = [_ if _ is None else torch.from_numpy(_).float().cuda() for _ in tmp]

        real_A_obj, real_A_rel, real_A_atr, real_B_obj, real_B_rel, real_B_atr = tmp

        iteration += 1

        fake_B_rel = netG_A_rel(real_A_rel)
        rec_A_rel = netG_B_rel(fake_B_rel)
        idt_B_rel = netG_B_rel(real_A_rel)

        fake_A_rel = netG_B_rel(real_B_rel)
        rec_B_rel = netG_A_rel(fake_A_rel)
        idt_A_rel = netG_A_rel(real_B_rel)

        # Obj
        fake_B_obj = netG_A_obj(real_A_obj)
        rec_A_obj = netG_B_obj(fake_B_obj)
        idt_B_obj = netG_B_obj(real_A_obj)

        fake_A_obj = netG_B_obj(real_B_obj)
        rec_B_obj = netG_A_obj(fake_A_obj)
        idt_A_obj = netG_A_obj(real_B_obj)

        # Atr
        fake_B_atr = netG_A_atr(real_A_atr)
        rec_A_atr = netG_B_atr(fake_B_atr)
        idt_B_atr = netG_B_atr(real_A_atr)

        fake_A_atr = netG_B_atr(real_B_atr)
        rec_B_atr = netG_A_atr(fake_A_atr)
        idt_A_atr = netG_A_atr(real_B_atr)

        domain_A = [real_A_obj, real_A_rel, real_A_atr,
                    fake_A_obj, fake_A_rel, fake_A_atr,
                    rec_A_obj, rec_A_rel, rec_A_atr,
                    idt_A_obj, idt_A_rel, idt_A_atr]
        domain_B = [real_B_obj, real_B_rel, real_B_atr,
                    fake_B_obj, fake_B_rel, fake_B_atr,
                    rec_B_obj, rec_B_rel, rec_B_atr,
                    idt_B_obj, idt_B_rel, idt_B_atr]
        # G_A and G_B
        utils.set_requires_grad([netD_A_obj, netD_A_rel, netD_A_atr, netD_B_obj, netD_B_rel, netD_B_atr], False)  # Ds require no gradients when optimizing Gs
        optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        loss_G = cycle_GAN_backward_G(opt, criterionGAN, criterionCycle, criterionIdt,
                                      netG_A_obj, netG_A_rel, netG_A_atr, netG_B_obj, netG_B_rel, netG_B_atr,
                                      netD_A_obj, netD_A_rel, netD_A_atr, netD_B_obj, netD_B_rel, netD_B_atr,
                                      domain_A, domain_B)
        loss_G.backward()
        optimizer_G.step()

        # D_A and D_B
        utils.set_requires_grad([netD_A_obj, netD_A_rel, netD_A_atr, netD_B_obj, netD_B_rel, netD_B_atr], True)
        optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        loss_D_A = cycle_GAN_backward_D(opt, fake_B_pool_obj, fake_B_pool_rel, fake_B_pool_atr,
                                        netD_A_obj, netD_A_rel, netD_A_atr, criterionGAN,
                                        real_B_obj, real_B_rel, real_B_atr, fake_B_obj, fake_B_rel, fake_B_atr)
        loss_D_A.backward()
        loss_D_B = cycle_GAN_backward_D(opt, fake_A_pool_obj, fake_A_pool_rel, fake_A_pool_atr,
                                        netD_B_obj, netD_B_rel, netD_B_atr, criterionGAN,
                                        real_A_obj, real_A_rel, real_A_atr, fake_A_obj, fake_A_rel, fake_A_atr)
        loss_D_B.backward()
        optimizer_D.step()  # update D_A and D_B's weights

        end = time.time()
        train_loss_G = loss_G.item()
        train_loss_D_A = loss_D_A.item()
        train_loss_D_B = loss_D_B.item()
        print("{}/{:.1f}/{}/{}|train_loss={:.3f}|train_loss_G={:.3f}|train_loss_D_A={:.3f}|train_loss_D_B={:.3f}|time/batch = {:.3f}".
              format(opt.id, train_precentage, iteration, epoch, train_loss, train_loss_G, train_loss_D_A, train_loss_D_B, end - start))
        torch.cuda.synchronize()

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0) and (iteration != 0):
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'train_loss_G', train_loss_G, iteration)
            add_summary_value(tb_summary_writer, 'train_loss_D_A', train_loss_D_A, iteration)
            add_summary_value(tb_summary_writer, 'train_loss_D_B', train_loss_D_B, iteration)

            # add hype parameters
            add_summary_value(tb_summary_writer, 'beam_size', opt.beam_size, iteration)
            add_summary_value(tb_summary_writer, 'lambdaA', opt.lambda_A, iteration)
            add_summary_value(tb_summary_writer, 'lambdaB', opt.lambda_B, iteration)
            add_summary_value(tb_summary_writer, 'pool_size', opt.pool_size, iteration)
            add_summary_value(tb_summary_writer, 'gan_type', opt.gan_type, iteration)
            add_summary_value(tb_summary_writer, 'gan_d_type', opt.gan_d_type, iteration)
            add_summary_value(tb_summary_writer, 'gan_g_type', opt.gan_g_type, iteration)


        if (iteration % opt.save_checkpoint_every == 0) and (iteration != 0):
            val_loss = eval_utils_gan.eval_split_gan(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, loader, loader_i2t)
            val_loss = val_loss.item()
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            current_score = - val_loss
            best_flag = False

            save_id = iteration / opt.save_checkpoint_every
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True

            checkpoint_path = os.path.join(opt.checkpoint_path, 'model_D.pth')
            torch.save({'epoch': epoch,
                        'netD_A_atr': netD_A_atr.state_dict(),
                        'netD_A_obj': netD_A_obj.state_dict(),
                        'netD_A_rel': netD_A_rel.state_dict(),
                        'netD_B_atr': netD_B_atr.state_dict(),
                        'netD_B_obj': netD_B_obj.state_dict(),
                        'netD_B_rel': netD_B_rel.state_dict()
                        }, checkpoint_path)

            checkpoint_path = os.path.join(opt.checkpoint_path, 'model_G.pth')
            torch.save({'epoch': epoch,
                        'netG_A_atr': netG_A_atr.state_dict(),
                        'netG_A_obj': netG_A_obj.state_dict(),
                        'netG_A_rel': netG_A_rel.state_dict(),
                        'netG_B_atr': netG_B_atr.state_dict(),
                        'netG_B_obj': netG_B_obj.state_dict(),
                        'netG_B_rel': netG_B_rel.state_dict()
                        }, checkpoint_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()

            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model_D-best.pth')
                torch.save({'epoch': epoch,
                            'netD_A_atr': netD_A_atr.state_dict(),
                            'netD_A_obj': netD_A_obj.state_dict(),
                            'netD_A_rel': netD_A_rel.state_dict(),
                            'netD_B_atr': netD_B_atr.state_dict(),
                            'netD_B_obj': netD_B_obj.state_dict(),
                            'netD_B_rel': netD_B_rel.state_dict()
                            }, checkpoint_path)

                checkpoint_path = os.path.join(opt.checkpoint_path, 'model_G-best.pth')
                torch.save({'epoch': epoch,
                            'netG_A_atr': netG_A_atr.state_dict(),
                            'netG_A_obj': netG_A_obj.state_dict(),
                            'netG_A_rel': netG_A_rel.state_dict(),
                            'netG_B_atr': netG_B_atr.state_dict(),
                            'netG_B_obj': netG_B_obj.state_dict(),
                            'netG_B_rel': netG_B_rel.state_dict()
                            }, checkpoint_path)

                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

        # Update the iteration and epoch
        if data['bounds']['wrapped']:
            # current_lr = update_learning_rate(schedulers, optimizers)
            epoch += 1
            update_lr_flag = True
            # make evaluation on validation set, and save model
            # lang_stats_isg = eval_utils_gan.eval_split_i2t(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, loader, loader_i2t)
            lang_stats_isg = eval_utils_gan.eval_split_g2t(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, loader, loader_i2t)

            if lang_stats_isg is not None:
                for k, v in lang_stats_isg.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()

opt.caption_model='sep_self_att_sep_gan_only'
opt.input_json='data/coco_cn/cocobu_gan_ssg.json'
opt.input_json_isg='data/coco_cn/cocobu_gan_isg.json'
opt.input_label_h5='data/coco_cn/cocobu_gan_isg_label.h5'
opt.ssg_dict_path='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'
opt.rela_dict_dir='data/rela_dict.npy'
opt.input_fc_dir='data/cocobu_fc'
opt.input_att_dir='data/cocobu_att'
opt.input_box_dir='data/cocotalk_box'
opt.input_label_h5='data/cocobu_label.h5'

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
train(opt)
