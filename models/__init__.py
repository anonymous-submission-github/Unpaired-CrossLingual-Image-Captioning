from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .NewFCModel import *
from .AttModel_SepSelfAtt_Sep_vv1 import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_p2 import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2 import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_p2 import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2 import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2 import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_en import *
from .AttModel_SepSelfAtt_Sep_vv1_RCSLS_en_p2 import *
from .AttModel_SepSelfAtt_Sep_vv1_p2 import *
from .AttModel_UP_SepSelfAtt_Sep import *
from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_p2 import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_p2 import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2 import *
from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2 import *
# from .AttModel_UP_SepSelfAtt_Sep_Extract import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_p2_Extract import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_p2_Extract import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_p2_Extract import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2_Extract import *
# from .AttModel_UP_SepSelfAtt_Sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2_Extract import *
from .weight_init import Model_init_en
from .weight_init import Model_init_zh
from .weight_init import Model_init

def setup(opt):
    
    if opt.caption_model == 'newfc':
        model = NewFCModel(opt)
    # DenseAtt
    elif opt.caption_model == 'up_gtssg_self_att_sep':
        model = (opt)
    elif opt.caption_model == 'sep_self_att_sep_gan_pseudo':
        model = GTssgModel_UP_SepSelfAtt_SEP(opt)
    elif opt.caption_model == 'up_gtssg_sep_self_att_sep': # final model
        model = GTssgModel_UP_SepSelfAtt_SEP(opt)
    elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_p2': # final model
        model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_p2(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_add_wordmap_p2': # final model
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_p2(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_add_wordmap_add_global_p2': # final model
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_extract':
    #     model = GTssgModel_UP_SepSelfAtt_SEP_Extract(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_p2_extract':
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_p2_Extract(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_p2_extract':
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_p2_Extract(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_p2_extract':
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_p2_Extract(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2_extract':
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2_Extract(opt)
    # elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2_extract':
    #     model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2_Extract(opt)
    elif opt.caption_model == 'up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2':
        model = GTssgModel_UP_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1':
        model = GTssgModel_SepSelfAtt_SEP_vv1(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_p2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_p2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_p2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_p2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_en':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_en(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_RCSLS_en_p2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_RCSLS_en_p2(opt)
    elif opt.caption_model == 'gtssg_sep_self_att_sep_vv1_p2':
        model = GTssgModel_SepSelfAtt_SEP_vv1_p2(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

   # # random initialize
   #
   #  save_id_real = getattr(opt, 'save_id', '')
   #  if save_id_real == '':
   #      save_id_real = opt.id
   #  model = Model_init(model, opt)
   #  return model


    if getattr(opt, 'p_flag', 0) == 1:
        save_id_real = getattr(opt, 'save_id', '')
        if save_id_real == '':
            save_id_real = opt.id_p
        # model = Model_init_en(model, opt)

        # check compatibility if training is continued from previously saved model
        if vars(opt).get('start_from_en', None) is not None:
            # check if all necessary files exist
            assert os.path.isdir(opt.checkpoint_path_p)," %s must be a a path" % opt.start_from
            # assert os.path.isfile(os.path.join(opt.checkpoint_path, 'infos.pkl')),"infos.pkl file does not exist in path %s"%opt.start_from
            try:
                model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path_p, 'model-best.pth')))
                print('Load from {}'.format(os.path.join(opt.checkpoint_path_p, 'model-best.pth')))
            except:
                model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path_p, 'model.pth')))
                print('Load from {}'.format(os.path.join(opt.checkpoint_path_p, 'model.pth')))
        else:
            model = Model_init_en(model, opt)
            # model = Model_init(model, opt)
    else:
        save_id_real = getattr(opt, 'save_id', '')
        if save_id_real == '':
            save_id_real = opt.id
        # model = Model_init_zh(model, opt)

        # check compatibility if training is continued from previously saved model
        if vars(opt).get('start_from', None) is not None:
            # check if all necessary files exist
            assert os.path.isdir(opt.checkpoint_path), " %s must be a a path" % opt.start_from
            # assert os.path.isfile(os.path.join(opt.checkpoint_path, 'infos.pkl')),"infos.pkl file does not exist in path %s"%opt.start_from
            try:
                model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
                print('Load from {}'.format(os.path.join(opt.start_from, 'model-best.pth')))
            except:
                model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
                print('Load from {}'.format(os.path.join(opt.start_from, 'model.pth')))
        else:
            model=Model_init_zh(model,opt)
            # model = Model_init(model, opt)
        # check compatibility if training is continued from previously saved model
    return model