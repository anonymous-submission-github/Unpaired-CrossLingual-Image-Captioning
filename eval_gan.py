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
# from dataloader_gan_paired import *
import eval_utils_gan
import misc.utils as utils
from misc.rewards_up import init_scorer, get_self_critical_reward
from gan_utils import *
import json

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

def eval_cap(opt):
    opt.start_from = None
    opt.start_from_gan = 'data/save_gan_rcsls_s_g_naacl/20201121_185728_sep_self_att_sep_gan_only'
    opt.checkpoint_path_gan = opt.start_from_gan
    opt.init_path_zh = 'data/save_for_finetune/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/model-best.pth'
    opt.input_isg_dir = "data/coco_graph_extract_ft_isg_joint_rcsls_submap_global_naacl_self_gate_finetune"
    opt.input_ssg_dir = "data/coco_graph_extract_ft_ssg_joint_rcsls_submap_global_naacl_self_gate_finetune"
    opt.caption_model_to_replace = 'up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2'

    opt.gpu=0

    opt.batch_size = 50
    opt.beam_size = 5
    opt.dump_path=1
    opt.caption_model = 'sep_self_att_sep_gan_only'
    opt.input_json = 'data/coco_cn/cocobu_gan_ssg.json'
    opt.input_json_isg = 'data/coco_cn/cocobu_gan_isg.json'
    opt.input_label_h5 = 'data/coco_cn/cocobu_gan_isg_label.h5'
    opt.ssg_dict_path = 'data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'
    opt.rela_dict_dir = 'data/rela_dict.npy'
    opt.input_fc_dir = 'data/cocobu_fc'
    opt.input_att_dir = 'data/cocobu_att'
    opt.input_box_dir = 'data/cocotalk_box'
    opt.input_label_h5 = 'data/cocobu_label.h5'


    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5
    opt.split = 'test'
    loader = DataLoader_GAN(opt)
    loader_i2t = DataLoader_UP(opt)

    opt.vocab_size = loader.vocab_size
    if opt.use_rela == 1:
        opt.rela_dict_size = loader.rela_dict_size
    opt.seq_length = loader.seq_length

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

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # opt.caption_model = 'up_gtssg_sep_self_att_sep'
    opt.caption_model = opt.caption_model_to_replace
    model = models.setup(opt).cuda()
    model.eval()

    opt.start_from = opt.start_from_gan
    opt.checkpoint_path = opt.start_from_gan
    opt.id = opt.checkpoint_path.split('/')[-1]
    start_from=opt.start_from_gan


    with open(os.path.join(opt.checkpoint_path, 'infos.pkl')) as f:
        infos = cPickle.load(f)
        saved_model_opt = infos['opt']
        saved_model_opt.start_from_gan = start_from
        saved_model_opt.checkpoint_path_gan = start_from

    netG_A_obj = GAN_init_G(saved_model_opt, Generator(saved_model_opt), type='netG_A_obj').cuda().eval()
    netG_A_rel = GAN_init_G(saved_model_opt, Generator(saved_model_opt), type='netG_A_rel').cuda().eval()
    netG_A_atr = GAN_init_G(saved_model_opt, Generator(saved_model_opt), type='netG_A_atr').cuda().eval()

    netG_B_obj = GAN_init_G(saved_model_opt, Generator(saved_model_opt), type='netG_B_obj').cuda().eval()
    netG_B_rel = GAN_init_G(saved_model_opt, Generator(saved_model_opt), type='netG_B_rel').cuda().eval()
    netG_B_atr = GAN_init_G(saved_model_opt, Generator(saved_model_opt), type='netG_B_atr').cuda().eval()

    val_loss, cache_path = eval_utils_gan.eval_split_gan_v2(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, netG_B_obj, netG_B_rel, netG_B_atr, loader, loader_i2t,opt.split)

opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
eval_cap(opt)