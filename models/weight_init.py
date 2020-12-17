from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import misc.utils as utils
import torch
from collections import OrderedDict

def freeze_param(model):
    for param in model.parameters():
        param.requires_grad = False

def activate_param(model):
    for param in model.parameters():
        param.requires_grad = True

def Model_init(model, opt):
    path=''
    print('Load from {}'.format(path))
    if len(path)>0:
        other = torch.load(path)
        try:
            model.load_state_dict(other)
            if getattr(opt, 'freeze_i2t', 0):
                freeze_param(model)
        except:
            for i in range(len(other.items())):
                if 'model.self_att.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att")
                    del model.self_att.projection[0].weight
                    del model.self_att.projection[0].bias
                    del model.self_att.projection[2].weight
                    del model.self_att.projection[3].bias
                    model.self_att.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att)
                if 'embed.0.weight' in other.items()[i][0] and 'fc_embed.0.weight' not in other.items()[i][0] and 'att_embed.0.weight' not in other.items()[i][0]:
                    print("  > Initialize embed")
                    del model.embed[0].weight
                    model.embed[0].weight = nn.Parameter(other.items()[i][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.embed)
                if 'fc_embed.0.weight' in other.items()[i][0]:
                    print("  > Initialize fc_embed")
                    del model.fc_embed[0].weight
                    del model.fc_embed[0].bias
                    model.fc_embed[0].weight = nn.Parameter(other.items()[i][1])
                    model.fc_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.fc_embed)
                if 'att_embed.0.weight' in other.items()[i][0]:
                    print("  > Initialize att_embed")
                    del model.att_embed[0].weight
                    del model.att_embed[0].bias
                    model.att_embed[0].weight = nn.Parameter(other.items()[i][1])
                    model.att_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.att_embed)
                if 'sbj_rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize sbj_rela_fc")
                    del model.sbj_rela_fc[0].weight
                    del model.sbj_rela_fc[0].bias
                    model.sbj_rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.sbj_rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_fc)
                if 'obj_rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_rela_fc")
                    del model.obj_rela_fc[0].weight
                    del model.obj_rela_fc[0].bias
                    model.obj_rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_fc)
                if 'obj_obj_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_obj_fc")
                    del model.obj_obj_fc[0].weight
                    del model.obj_obj_fc[0].bias
                    model.obj_obj_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_obj_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_fc)
                if 'obj_attr_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_attr_fc")
                    del model.obj_attr_fc[0].weight
                    del model.obj_attr_fc[0].bias
                    model.obj_attr_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_attr_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_attr_fc)
                if 'rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize rela_fc")
                    del model.rela_fc[0].weight
                    del model.rela_fc[0].bias
                    model.rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.rela_fc)
                if 'attr_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize attr_fc")
                    del model.attr_fc[0].weight
                    del model.attr_fc[0].bias
                    model.attr_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.attr_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.attr_fc)
                if 'logit.weight' in other.items()[i][0]:
                    print("  > Initialize logit")
                    del model.logit.weight
                    del model.logit.bias
                    model.logit.weight = nn.Parameter(other.items()[i][1])
                    model.logit.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.logit)
                if 'ctx2att.weight' in other.items()[i][0]:
                    print("  > Initialize ctx2att")
                    del model.ctx2att.weight
                    del model.ctx2att.bias
                    model.ctx2att.weight = nn.Parameter(other.items()[i][1])
                    model.ctx2att.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.ctx2att)
                if 'core.mapping_triplet.0.weight' in other.items()[i][0] and False:
                    print("  > Initialize mapping_triplet")
                    del model.core.mapping_triplet[0].weight
                    del model.core.mapping_triplet[0].bias
                    model.core.mapping_triplet[0].weight = nn.Parameter(other.items()[i][1])
                    model.core.mapping_triplet[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.mapping_triplet)
                if 'core.lstm1.weight_ih' in other.items()[i][0]:
                    print("  > Initialize core")
                    del model.core.lstm1.weight_ih
                    del model.core.lstm1.weight_hh
                    del model.core.lstm1.bias_ih
                    del model.core.lstm1.bias_hh
                    del model.core.lstm2.weight_ih
                    del model.core.lstm2.weight_hh
                    del model.core.lstm2.bias_ih
                    del model.core.lstm2.bias_hh
                    model.core.lstm1.weight_ih = nn.Parameter(other.items()[i][1])
                    model.core.lstm1.weight_hh = nn.Parameter(other.items()[i + 1][1])
                    model.core.lstm1.bias_ih = nn.Parameter(other.items()[i + 2][1])
                    model.core.lstm1.bias_hh = nn.Parameter(other.items()[i + 3][1])
                    model.core.lstm2.weight_ih = nn.Parameter(other.items()[i + 4][1])
                    model.core.lstm2.weight_hh = nn.Parameter(other.items()[i + 5][1])
                    model.core.lstm2.bias_ih = nn.Parameter(other.items()[i + 6][1])
                    model.core.lstm2.bias_hh = nn.Parameter(other.items()[i + 7][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.lstm1)
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.lstm2)
                if 'model.core.attention.h2att.weight' in other.items()[i][0]:
                    print("  > Initialize core.attention.h2att")
                    del model.core.attention.h2att.weight
                    del model.core.attention.h2att.bias
                    del model.core.attention.alpha_net.weight
                    del model.core.attention.alpha_net.bias
                    model.core.attention.h2att.weight = nn.Parameter(other.items()[i][1])
                    model.core.attention.h2att.bias = nn.Parameter(other.items()[i + 1][1])
                    model.core.attention.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
                    model.core.attention.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.attention)
    return model


def Model_init_zh(model, opt):
    # path=''
    path = opt.init_path_zh
    print('Load from {}'.format(path))
    if len(path)>0:
        other = torch.load(path)
        try:
            model.load_state_dict(other,strict=False)
            # model.load_state_dict(other)
            if getattr(opt, 'freeze_i2t', 0):
                freeze_param(model)
        except:
            for i in range(len(other.items())):
                # print (other.items()[i][0])
                if 'map_emb_dim.weight' in other.items()[i][0]:
                    print("  > Initialize map_emb_dim")
                    del model.map_emb_dim.weight
                    del model.map_emb_dim.bias
                    model.map_emb_dim.weight= nn.Parameter(other.items()[i][1])
                    model.map_emb_dim.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.map_emb_dim)
                if 'en2zh_maping.0.weight' in other.items()[i][0]:
                    print("  > Initialize en2zh_maping")
                    del model.en2zh_maping[0].weight
                    del model.en2zh_maping[0].bias
                    model.en2zh_maping[0].weight= nn.Parameter(other.items()[i][1])
                    model.en2zh_maping[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.en2zh_maping)

                if 'graph_weights' in other.items()[i][0]:
                    print("  > Initialize graph_weights")
                    del model.graph_weights
                    model.graph_weights= nn.Parameter(other.items()[i][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.graph_weights)

                if 'mapping_hierachy.weight' in other.items()[i][0]:
                    print("  > Initialize mapping_hierachy")
                    del model.mapping_hierachy.weight
                    del model.mapping_hierachy.bias
                    model.mapping_hierachy.weight= nn.Parameter(other.items()[i][1])
                    model.mapping_hierachy.bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.mapping_hierachy)


                if 'mlp.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize mlp")
                    del model.mlp.projection[0].weight
                    del model.mlp.projection[0].bias
                    del model.mlp.projection[2].weight
                    del model.mlp.projection[2].bias
                    del model.mlp.projection[4].weight
                    del model.mlp.projection[4].bias

                    model.mlp.projection[0].weight= nn.Parameter(other.items()[i][1])
                    model.mlp.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.mlp.projection[2].weight= nn.Parameter(other.items()[i + 2][1])
                    model.mlp.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    model.mlp.projection[4].weight= nn.Parameter(other.items()[i + 4][1])
                    model.mlp.projection[4].bias = nn.Parameter(other.items()[i + 5][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.mapping_hierachy)


                if 'self_att_obj_en.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att_obj_en")
                    del model.self_att_obj_en.projection[0].weight
                    del model.self_att_obj_en.projection[0].bias
                    del model.self_att_obj_en.projection[2].weight
                    del model.self_att_obj_en.projection[2].bias
                    model.self_att_obj_en.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_obj_en.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_obj_en.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_obj_en.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_obj_en)

                if 'self_att_obj_w.projection.0.weight' in other.items()[i][0]:
                    del model.self_att_obj_w.projection[0].weight
                    del model.self_att_obj_w.projection[0].bias
                    del model.self_att_obj_w.projection[2].weight
                    del model.self_att_obj_w.projection[2].bias
                    model.self_att_obj_w.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_obj_w.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_obj_w.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_obj_w.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_obj_w)

                if 'self_att_obj_s.projection.0.weight' in other.items()[i][0]:
                    del model.self_att_obj_s.projection[0].weight
                    del model.self_att_obj_s.projection[0].bias
                    del model.self_att_obj_s.projection[2].weight
                    del model.self_att_obj_s.projection[2].bias
                    model.self_att_obj_s.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_obj_s.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_obj_s.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_obj_s.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_obj_s)

                if 'self_att_obj_g.projection.0.weight' in other.items()[i][0]:
                    del model.self_att_obj_g.projection[0].weight
                    del model.self_att_obj_g.projection[0].bias
                    del model.self_att_obj_g.projection[2].weight
                    del model.self_att_obj_g.projection[2].bias
                    model.self_att_obj_g.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_obj_g.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_obj_g.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_obj_g.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_obj_g)

                if 'self_att_rel.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att_rel")
                    del model.self_att_rel.projection[0].weight
                    del model.self_att_rel.projection[0].bias
                    del model.self_att_rel.projection[2].weight
                    del model.self_att_rel.projection[2].bias
                    model.self_att_rel.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_rel.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_rel.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_rel.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_rel)
                if 'self_att_atr.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att_atr")
                    del model.self_att_atr.projection[0].weight
                    del model.self_att_atr.projection[0].bias
                    del model.self_att_atr.projection[2].weight
                    del model.self_att_atr.projection[2].bias
                    model.self_att_atr.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_atr.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_atr.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_atr.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_atr)
                if 'embed.0.weight' in other.items()[i][0] and 'fc_embed.0.weight' not in other.items()[i][0] and 'att_embed.0.weight' not in other.items()[i][0]:
                    print("  > Initialize embed")
                    del model.embed[0].weight
                    model.embed[0].weight = nn.Parameter(other.items()[i][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.embed)
                if 'fc_embed.0.weight' in other.items()[i][0]:
                    print("  > Initialize fc_embed")
                    del model.fc_embed[0].weight
                    del model.fc_embed[0].bias
                    model.fc_embed[0].weight = nn.Parameter(other.items()[i][1])
                    model.fc_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.fc_embed)
                if 'att_embed.0.weight' in other.items()[i][0]:
                    print("  > Initialize att_embed")
                    del model.att_embed[0].weight
                    del model.att_embed[0].bias
                    model.att_embed[0].weight = nn.Parameter(other.items()[i][1])
                    model.att_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.att_embed)
                if 'sbj_rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize sbj_rela_fc")
                    del model.sbj_rela_fc[0].weight
                    del model.sbj_rela_fc[0].bias
                    model.sbj_rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.sbj_rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_fc)

                # if 'sbj_rela_fc_map.0.weight' in other.items()[i][0]:
                #     print("  > Initialize sbj_rela_fc_map")
                #     del model.sbj_rela_fc_map[0].weight
                #     del model.sbj_rela_fc_map[0].bias
                #     model.sbj_rela_fc_map[0].weight = nn.Parameter(other.items()[i][1])
                #     model.sbj_rela_fc_map[0].bias = nn.Parameter(other.items()[i + 1][1])
                #     if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_fc_map)
                if 'sbj_rela_v2_map_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize sbj_rela_v2_map_fc")
                    del model.sbj_rela_v2_map_fc[0].weight
                    del model.sbj_rela_v2_map_fc[0].bias
                    model.sbj_rela_v2_map_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.sbj_rela_v2_map_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_v2_map_fc)
                if 'obj_rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_rela_fc")
                    del model.obj_rela_fc[0].weight
                    del model.obj_rela_fc[0].bias
                    model.obj_rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_fc)

                # if 'obj_rela_fc_map.0.weight' in other.items()[i][0]:
                #     print("  > Initialize obj_rela_fc_map")
                #     del model.obj_rela_fc_map[0].weight
                #     del model.obj_rela_fc_map[0].bias
                #     model.obj_rela_fc_map[0].weight = nn.Parameter(other.items()[i][1])
                #     model.obj_rela_fc_map[0].bias = nn.Parameter(other.items()[i + 1][1])
                #     if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_fc_map)
                if 'obj_rela_v2_map_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_rela_v2_map_fc")
                    del model.obj_rela_v2_map_fc[0].weight
                    del model.obj_rela_v2_map_fc[0].bias
                    model.obj_rela_v2_map_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_rela_v2_map_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_v2_map_fc)
                if 'obj_obj_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_obj_fc")
                    del model.obj_obj_fc[0].weight
                    del model.obj_obj_fc[0].bias
                    model.obj_obj_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_obj_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_fc)

                # if 'obj_obj_fc_map.0.weight' in other.items()[i][0]:
                #     print("  > Initialize obj_obj_fc_map")
                #     del model.obj_obj_fc_map[0].weight
                #     del model.obj_obj_fc_map[0].bias
                #     model.obj_obj_fc_map[0].weight = nn.Parameter(other.items()[i][1])
                #     model.obj_obj_fc_map[0].bias = nn.Parameter(other.items()[i + 1][1])
                #     if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_fc_map)
                if 'obj_obj_v2_map_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_obj_v2_map_fc")
                    del model.obj_obj_v2_map_fc[0].weight
                    del model.obj_obj_v2_map_fc[0].bias
                    model.obj_obj_v2_map_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_obj_v2_map_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_v2_map_fc)
                if 'obj_attr_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_attr_fc")
                    del model.obj_attr_fc[0].weight
                    del model.obj_attr_fc[0].bias
                    model.obj_attr_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_attr_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    # if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_attr_fc)
                if 'rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize rela_fc")
                    del model.rela_fc[0].weight
                    del model.rela_fc[0].bias
                    model.rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.rela_fc)
                if 'attr_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize attr_fc")
                    del model.attr_fc[0].weight
                    del model.attr_fc[0].bias
                    model.attr_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.attr_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.attr_fc)
                if 'logit.weight' in other.items()[i][0]:
                    print("  > Initialize logit")
                    del model.logit.weight
                    del model.logit.bias
                    model.logit.weight = nn.Parameter(other.items()[i][1])
                    model.logit.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.logit)
                if 'ctx2att.weight' in other.items()[i][0]:
                    print("  > Initialize ctx2att")
                    del model.ctx2att.weight
                    del model.ctx2att.bias
                    model.ctx2att.weight = nn.Parameter(other.items()[i][1])
                    model.ctx2att.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.ctx2att)
                if 'core.mapping_triplet.0.weight' in other.items()[i][0] and False:
                    print("  > Initialize mapping_triplet")
                    del model.core.mapping_triplet[0].weight
                    del model.core.mapping_triplet[0].bias
                    model.core.mapping_triplet[0].weight = nn.Parameter(other.items()[i][1])
                    model.core.mapping_triplet[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.mapping_triplet)
                if 'core.lstm1.weight_ih' in other.items()[i][0]:
                    print("  > Initialize core")
                    del model.core.lstm1.weight_ih
                    del model.core.lstm1.weight_hh
                    del model.core.lstm1.bias_ih
                    del model.core.lstm1.bias_hh
                    del model.core.lstm2.weight_ih
                    del model.core.lstm2.weight_hh
                    del model.core.lstm2.bias_ih
                    del model.core.lstm2.bias_hh
                    model.core.lstm1.weight_ih = nn.Parameter(other.items()[i][1])
                    model.core.lstm1.weight_hh = nn.Parameter(other.items()[i + 1][1])
                    model.core.lstm1.bias_ih = nn.Parameter(other.items()[i + 2][1])
                    model.core.lstm1.bias_hh = nn.Parameter(other.items()[i + 3][1])
                    model.core.lstm2.weight_ih = nn.Parameter(other.items()[i + 4][1])
                    model.core.lstm2.weight_hh = nn.Parameter(other.items()[i + 5][1])
                    model.core.lstm2.bias_ih = nn.Parameter(other.items()[i + 6][1])
                    model.core.lstm2.bias_hh = nn.Parameter(other.items()[i + 7][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.lstm1)
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.lstm2)
                if 'model.core.attention.h2att.weight' in other.items()[i][0]:
                    print("  > Initialize core.attention.h2att")
                    del model.core.attention.h2att.weight
                    del model.core.attention.h2att.bias
                    del model.core.attention.alpha_net.weight
                    del model.core.attention.alpha_net.bias
                    model.core.attention.h2att.weight = nn.Parameter(other.items()[i][1])
                    model.core.attention.h2att.bias = nn.Parameter(other.items()[i + 1][1])
                    model.core.attention.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
                    model.core.attention.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.attention)
    return model

def Model_init_en(model, opt):
    path=''
    # path = opt.init_path_en
    # print('Load from {}'.format(path))
    if len(path)>0:
        other = torch.load(path)
        try:
            model.load_state_dict(other)
            if getattr(opt, 'freeze_i2t', 0):
                freeze_param(model)
        except:
            for i in range(len(other.items())):
                if 'map_emb_dim.weight' in other.items()[i][0]:
                    print("  > Initialize map_emb_dim")
                    del model.map_emb_dim.weight
                    del model.map_emb_dim.bias
                    model.map_emb_dim.weight = nn.Parameter(other.items()[i][1])
                    model.map_emb_dim.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.map_emb_dim)
                if 'en2zh_maping.0.weight' in other.items()[i][0]:
                    print("  > Initialize en2zh_maping")
                    del model.en2zh_maping[0].weight
                    del model.en2zh_maping[0].bias
                    model.en2zh_maping[0].weight = nn.Parameter(other.items()[i][1])
                    model.en2zh_maping[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.en2zh_maping)
                if 'self_att_obj.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att_obj")
                    del model.self_att_obj.projection[0].weight
                    del model.self_att_obj.projection[0].bias
                    del model.self_att_obj.projection[2].weight
                    del model.self_att_obj.projection[2].bias
                    model.self_att_obj.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_obj.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_obj.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_obj.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_obj)
                if 'self_att_rel.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att_rel")
                    del model.self_att_rel.projection[0].weight
                    del model.self_att_rel.projection[0].bias
                    del model.self_att_rel.projection[2].weight
                    del model.self_att_rel.projection[2].bias
                    model.self_att_rel.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_rel.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_rel.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_rel.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_rel)
                if 'self_att_atr.projection.0.weight' in other.items()[i][0]:
                    print("  > Initialize self_att_atr")
                    del model.self_att_atr.projection[0].weight
                    del model.self_att_atr.projection[0].bias
                    del model.self_att_atr.projection[2].weight
                    del model.self_att_atr.projection[2].bias
                    model.self_att_atr.projection[0].weight = nn.Parameter(other.items()[i][1])
                    model.self_att_atr.projection[0].bias = nn.Parameter(other.items()[i + 1][1])
                    model.self_att_atr.projection[2].weight = nn.Parameter(other.items()[i + 2][1])
                    model.self_att_atr.projection[2].bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.self_att_atr)
                if 'embed.0.weight' in other.items()[i][0] and 'fc_embed.0.weight' not in other.items()[i][
                    0] and 'att_embed.0.weight' not in other.items()[i][0]:
                    print("  > Initialize embed")
                    del model.embed[0].weight
                    model.embed[0].weight = nn.Parameter(other.items()[i][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.embed)
                if 'fc_embed.0.weight' in other.items()[i][0]:
                    print("  > Initialize fc_embed")
                    del model.fc_embed[0].weight
                    del model.fc_embed[0].bias
                    model.fc_embed[0].weight = nn.Parameter(other.items()[i][1])
                    model.fc_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.fc_embed)
                if 'att_embed.0.weight' in other.items()[i][0]:
                    print("  > Initialize att_embed")
                    del model.att_embed[0].weight
                    del model.att_embed[0].bias
                    model.att_embed[0].weight = nn.Parameter(other.items()[i][1])
                    model.att_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.att_embed)
                if 'sbj_rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize sbj_rela_fc")
                    del model.sbj_rela_fc[0].weight
                    del model.sbj_rela_fc[0].bias
                    model.sbj_rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.sbj_rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_fc)
                    # if 'sbj_rela_fc_map.0.weight' in other.items()[i][0]:
                    #     print("  > Initialize sbj_rela_fc_map")
                    #     del model.sbj_rela_fc_map[0].weight
                    #     del model.sbj_rela_fc_map[0].bias
                    #     model.sbj_rela_fc_map[0].weight = nn.Parameter(other.items()[i][1])
                    #     model.sbj_rela_fc_map[0].bias = nn.Parameter(other.items()[i + 1][1])
                    #     if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_fc_map)
                if 'sbj_rela_v2_map_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize sbj_rela_v2_map_fc")
                    del model.sbj_rela_v2_map_fc[0].weight
                    del model.sbj_rela_v2_map_fc[0].bias
                    model.sbj_rela_v2_map_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.sbj_rela_v2_map_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.sbj_rela_v2_map_fc)
                if 'obj_rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_rela_fc")
                    del model.obj_rela_fc[0].weight
                    del model.obj_rela_fc[0].bias
                    model.obj_rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_fc)
                    # if 'obj_rela_fc_map.0.weight' in other.items()[i][0]:
                    #     print("  > Initialize obj_rela_fc_map")
                    #     del model.obj_rela_fc_map[0].weight
                    #     del model.obj_rela_fc_map[0].bias
                    #     model.obj_rela_fc_map[0].weight = nn.Parameter(other.items()[i][1])
                    #     model.obj_rela_fc_map[0].bias = nn.Parameter(other.items()[i + 1][1])
                    #     if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_fc_map)
                if 'obj_rela_v2_map_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_rela_v2_map_fc")
                    del model.obj_rela_v2_map_fc[0].weight
                    del model.obj_rela_v2_map_fc[0].bias
                    model.obj_rela_v2_map_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_rela_v2_map_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_rela_v2_map_fc)
                if 'obj_obj_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_obj_fc")
                    del model.obj_obj_fc[0].weight
                    del model.obj_obj_fc[0].bias
                    model.obj_obj_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_obj_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_fc)
                    # if 'obj_obj_fc_map.0.weight' in other.items()[i][0]:
                    #     print("  > Initialize obj_obj_fc_map")
                    #     del model.obj_obj_fc_map[0].weight
                    #     del model.obj_obj_fc_map[0].bias
                    #     model.obj_obj_fc_map[0].weight = nn.Parameter(other.items()[i][1])
                    #     model.obj_obj_fc_map[0].bias = nn.Parameter(other.items()[i + 1][1])
                    #     if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_fc_map)
                if 'obj_obj_v2_map_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_obj_v2_map_fc")
                    del model.obj_obj_v2_map_fc[0].weight
                    del model.obj_obj_v2_map_fc[0].bias
                    model.obj_obj_v2_map_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_obj_v2_map_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_obj_v2_map_fc)
                if 'obj_attr_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize obj_attr_fc")
                    del model.obj_attr_fc[0].weight
                    del model.obj_attr_fc[0].bias
                    model.obj_attr_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.obj_attr_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.obj_attr_fc)
                if 'rela_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize rela_fc")
                    del model.rela_fc[0].weight
                    del model.rela_fc[0].bias
                    model.rela_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.rela_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.rela_fc)
                if 'attr_fc.0.weight' in other.items()[i][0]:
                    print("  > Initialize attr_fc")
                    del model.attr_fc[0].weight
                    del model.attr_fc[0].bias
                    model.attr_fc[0].weight = nn.Parameter(other.items()[i][1])
                    model.attr_fc[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.attr_fc)
                if 'logit.weight' in other.items()[i][0]:
                    print("  > Initialize logit")
                    del model.logit.weight
                    del model.logit.bias
                    model.logit.weight = nn.Parameter(other.items()[i][1])
                    model.logit.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.logit)
                if 'ctx2att.weight' in other.items()[i][0]:
                    print("  > Initialize ctx2att")
                    del model.ctx2att.weight
                    del model.ctx2att.bias
                    model.ctx2att.weight = nn.Parameter(other.items()[i][1])
                    model.ctx2att.bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.ctx2att)
                if 'core.mapping_triplet.0.weight' in other.items()[i][0] and False:
                    print("  > Initialize mapping_triplet")
                    del model.core.mapping_triplet[0].weight
                    del model.core.mapping_triplet[0].bias
                    model.core.mapping_triplet[0].weight = nn.Parameter(other.items()[i][1])
                    model.core.mapping_triplet[0].bias = nn.Parameter(other.items()[i + 1][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.mapping_triplet)
                if 'core.lstm1.weight_ih' in other.items()[i][0]:
                    print("  > Initialize core")
                    del model.core.lstm1.weight_ih
                    del model.core.lstm1.weight_hh
                    del model.core.lstm1.bias_ih
                    del model.core.lstm1.bias_hh
                    del model.core.lstm2.weight_ih
                    del model.core.lstm2.weight_hh
                    del model.core.lstm2.bias_ih
                    del model.core.lstm2.bias_hh
                    model.core.lstm1.weight_ih = nn.Parameter(other.items()[i][1])
                    model.core.lstm1.weight_hh = nn.Parameter(other.items()[i + 1][1])
                    model.core.lstm1.bias_ih = nn.Parameter(other.items()[i + 2][1])
                    model.core.lstm1.bias_hh = nn.Parameter(other.items()[i + 3][1])
                    model.core.lstm2.weight_ih = nn.Parameter(other.items()[i + 4][1])
                    model.core.lstm2.weight_hh = nn.Parameter(other.items()[i + 5][1])
                    model.core.lstm2.bias_ih = nn.Parameter(other.items()[i + 6][1])
                    model.core.lstm2.bias_hh = nn.Parameter(other.items()[i + 7][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.lstm1)
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.lstm2)
                if 'model.core.attention.h2att.weight' in other.items()[i][0]:
                    print("  > Initialize core.attention.h2att")
                    del model.core.attention.h2att.weight
                    del model.core.attention.h2att.bias
                    del model.core.attention.alpha_net.weight
                    del model.core.attention.alpha_net.bias
                    model.core.attention.h2att.weight = nn.Parameter(other.items()[i][1])
                    model.core.attention.h2att.bias = nn.Parameter(other.items()[i + 1][1])
                    model.core.attention.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
                    model.core.attention.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
                    if getattr(opt, 'freeze_i2t', 0): freeze_param(model.core.attention)
    return model
