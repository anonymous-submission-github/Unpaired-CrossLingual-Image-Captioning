# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel
from .spectral_normalization import SpectralNorm

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.gan_type = getattr(opt, 'gan_type', 0)
        self.index_eval = getattr(opt, 'index_eval', 0)

        self.enable_i2t = getattr(opt, 'enable_i2t', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.use_attr_info = getattr(opt, 'use_attr_info', 1)

        # whether use relationship or not
        self.use_rela = getattr(opt, 'use_rela', 0)
        self.use_gru = getattr(opt, 'use_gru', 0)
        self.use_gfc = getattr(opt, 'use_gfc', 0)
        self.gru_t = getattr(opt, 'gru_t', 1)

        self.rbm_logit = getattr(opt, 'rbm_logit', 0)
        self.rbm_size = getattr(opt, 'rbm_size', 2000)

        # whether use sentence scene graph or not
        self.use_ssg = getattr(opt, 'use_ssg', 0)
        self.use_isg = getattr(opt, 'use_isg', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        self.self_att_obj = SelfAttention(opt)
        self.self_att_rel = SelfAttention(opt)
        self.self_att_atr = SelfAttention(opt)

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(inplace=True),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        if self.use_rela:
            self.rela_dict_len = getattr(opt, 'rela_dict_size', 0)

            if self.use_gru:
                self.rela_embed = nn.Embedding(self.rela_dict_len, self.rnn_size)
                self.gru = nn.GRU(self.rnn_size * 2, self.rnn_size)
            if self.use_gfc:
                self.rela_embed = nn.Linear(self.rela_dict_len, self.rnn_size, bias=False)
                self.sbj_fc = nn.Sequential(nn.Linear(self.rnn_size * 3, self.rnn_size),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(self.drop_prob_lm))
                self.obj_fc = nn.Sequential(nn.Linear(self.rnn_size * 3, self.rnn_size),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(self.drop_prob_lm))
                self.rela_fc = nn.Sequential(nn.Linear(self.rnn_size * 3, self.rnn_size),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm))
                self.attr_fc = nn.Sequential(nn.Linear(self.rnn_size * 2, self.rnn_size),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm))
        if self.use_ssg:
            self.sbj_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 3, self.rnn_size),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm))
            self.obj_rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 3, self.rnn_size),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm))
            self.obj_obj_fc = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(self.drop_prob_lm))
            self.obj_attr_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 2, self.rnn_size),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(self.drop_prob_lm))
            self.rela_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 3, self.rnn_size),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(self.drop_prob_lm))
            self.attr_fc = nn.Sequential(nn.Linear(self.input_encoding_size * 2, self.rnn_size),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(self.drop_prob_lm))

        if self.rbm_logit == 1:
            self.logit = Log_Rbm(self.rnn_size, self.rbm_size, self.vocab_size + 1)
        else:
            self.logit_layers = getattr(opt, 'logit_layers', 1)
            if self.logit_layers == 1:
                self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
            else:
                self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
                self.logit = nn.Sequential(*(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats

    def graph_gru(self, rela_data, pre_hidden_state):
        """
        :param att_feats: roi features of each bounding box
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param pre_hidden_state: previous hidden state
        :return: hidden_state: current hidden state
        """
        att_feats = rela_data['att_feats']
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        neighbor = torch.zeros([N_img, att_feats_size[1], att_feats_size[2]])
        neighbor = neighbor.cuda()
        # hidden_state = torch.zeros(pre_hidden_state.size())
        hidden_state = att_feats.clone()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_rela = np.int32(N_rela)
            box_num = np.zeros([att_feats_size[1], ])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id * seq_per_img, i, 0]
                obj_id = rela_matrix[img_id * seq_per_img, i, 1]
                rela_id = rela_matrix[img_id * seq_per_img, i, 2]
                sub_id = np.int32(sub_id)
                obj_id = np.int32(obj_id)
                rela_embedding = self.rela_embed(rela_id.long())
                rela_embedding = torch.squeeze(rela_embedding)
                neighbor[img_id, sub_id] += rela_embedding * att_feats[img_id * seq_per_img, obj_id]
                box_num[sub_id] += 1
            for i in range(att_feats_size[1]):
                if box_num[i] != 0:
                    neighbor[img_id, i] /= box_num[i]
                    input = torch.cat((att_feats[img_id * seq_per_img, i], neighbor[img_id, i]))
                    input = torch.unsqueeze(input, 0)
                    input = torch.unsqueeze(input, 0)
                    hidden_state_temp = pre_hidden_state[img_id * seq_per_img, i]
                    hidden_state_temp = torch.unsqueeze(hidden_state_temp, 0)
                    hidden_state_temp = torch.unsqueeze(hidden_state_temp, 0)
                    hidden_state_temp, out_temp = self.gru(input, hidden_state_temp)
                    hidden_state[img_id * seq_per_img:(img_id + 1) * seq_per_img, i] = torch.squeeze(hidden_state_temp)

        return hidden_state

    def graph_gfc(self, rela_data):
        """
        :param att_feats: roi features of each bounding box, [N_img*5, N_att_max, rnn_size]
        :param rela_feats: the embeddings of relationship, [N_img*5, N_rela_max, rnn_size]
        :param rela_matrix: relationship matrix, [N_img*5, N_rela_max, 3], N_img
                            is the batch size, N_rela_max is the maximum number
                            of relationship in rela_matrix.
        :param rela_masks: relationship masks, [N_img*5, N_rela_max].
                            For each row, the sum of that row is the total number
                            of realtionship.
        :param att_masks: attention masks, [N_img*5, N_att_max].
                            For each row, the sum of that row is the total number
                            of roi poolings.
        :param attr_matrix: attribute matrix,[N_img*5, N_attr_max, N_attr_each_max]
                            N_img is the batch size, N_attr_max is the maximum number
                            of attributes of one mini-batch, N_attr_each_max is the
                            maximum number of attributes of each objects in that mini-batch
        :param attr_masks: attribute masks, [N_img*5, N_attr_max, N_attr_each_max]
                            the sum of attr_masks[img_id*5,:,0] is the number of objects
                            which own attributes, the sum of attr_masks[img_id*5, obj_id, :]
                            is the number of attribute that object has
        :return: att_feats_new: new roi features
                 rela_feats_new: new relationship embeddings
                 attr_feats_new: new attribute features
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_matrix = rela_data['rela_matrix']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        attr_matrix = rela_data['attr_matrix']
        attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        rela_feats_size = rela_feats.size()
        attr_masks_size = attr_masks.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        att_feats_new = att_feats.clone()
        rela_feats_new = rela_feats.clone()
        if self.use_attr_info == 1:
            attr_feats_new = torch.zeros([attr_masks_size[0], attr_masks_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_box = torch.sum(att_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            N_box = int(N_box)
            box_num = np.ones([N_box, ])
            rela_num = np.ones([N_rela, ])
            for i in range(N_rela):
                sub_id = rela_matrix[img_id * seq_per_img, i, 0]
                sub_id = int(sub_id)
                box_num[sub_id] += 1.0
                obj_id = rela_matrix[img_id * seq_per_img, i, 1]
                obj_id = int(obj_id)
                box_num[obj_id] += 1.0
                rela_id = i
                rela_num[rela_id] += 1.0
                sub_feat_use = att_feats[img_id * seq_per_img, sub_id, :]
                obj_feat_use = att_feats[img_id * seq_per_img, obj_id, :]
                rela_feat_use = rela_feats[img_id * seq_per_img, rela_id, :]

                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, sub_id, :] += \
                    self.sbj_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, obj_id, :] += \
                    self.obj_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, rela_id, :] += \
                    self.rela_fc(torch.cat((sub_feat_use, obj_feat_use, rela_feat_use)))

            if self.use_attr_info == 1:
                N_obj_attr = torch.sum(attr_masks[img_id * seq_per_img, :, 0])
                N_obj_attr = int(N_obj_attr)
                for i in range(N_obj_attr):
                    attr_obj_id = int(attr_matrix[img_id * seq_per_img, i, 0])
                    obj_feat_use = att_feats[img_id * seq_per_img, int(attr_obj_id), :]
                    N_attr_each = torch.sum(attr_masks[img_id * seq_per_img, i, :])
                    for j in range(N_attr_each - 1):
                        attr_index = attr_matrix[img_id * seq_per_img, i, j + 1]
                        attr_one_hot = torch.zeros([self.rela_dict_len, ])
                        attr_one_hot = attr_one_hot.scatter_(0, attr_index.cpu().long(), 1).cuda()
                        attr_feat_use = self.rela_embed(attr_one_hot)
                        attr_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, i, :] += \
                            self.attr_fc(torch.cat((attr_feat_use, obj_feat_use)))
                    attr_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, i, :] = \
                        attr_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, i, :] / (float(N_attr_each) - 1)

            for i in range(N_box):
                att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i] = \
                    att_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i] / box_num[i]
            for i in range(N_rela):
                rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] = \
                    rela_feats_new[img_id * seq_per_img: (img_id + 1) * seq_per_img, i, :] / rela_num[i]

        rela_data['att_feats'] = att_feats_new
        rela_data['rela_feats'] = rela_feats_new
        if self.use_attr_info == 1:
            rela_data['attr_feats'] = attr_feats_new
        return rela_data

    def prepare_rela_feats(self, rela_data):
        """
        Change relationship index (one-hot) to relationship features, or change relationship
        probability to relationship features.
        :param rela_matrix:
        :param rela_masks:
        :return: rela_features, [N_img*5, N_rela_max, rnn_size]
        """
        rela_matrix = rela_data['rela_matrix']
        rela_masks = rela_data['rela_masks']

        rela_feats_size = rela_matrix.size()
        N_att = rela_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        rela_feats = torch.zeros([rela_feats_size[0], rela_feats_size[1], self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = torch.sum(rela_masks[img_id * seq_per_img, :])
            N_rela = int(N_rela)
            if N_rela > 0:
                rela_index = rela_matrix[img_id * seq_per_img, :N_rela, 2].cpu().long()
                rela_index = torch.unsqueeze(rela_index, 1)
                rela_one_hot = torch.zeros([N_rela, self.rela_dict_len])
                rela_one_hot = rela_one_hot.scatter_(1, rela_index, 1).cuda()
                rela_feats_temp = self.rela_embed(rela_one_hot)
                rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_rela, :] = rela_feats_temp
        rela_data['rela_feats'] = rela_feats
        return rela_data

    def merge_rela_att(self, rela_data):
        """
        merge attention features (roi features) and relationship features together
        :param att_feats: [N_att, N_att_max, rnn_size]
        :param att_masks: [N_att, N_att_max]
        :param rela_feats: [N_att, N_rela_max, rnn_size]
        :param rela_masks: [N_att, N_rela_max]
        :return: att_feats_new: [N_att, N_att_new_max, rnn_size]
                 att_masks_new: [N_att, N_att_new_max]
        """
        att_feats = rela_data['att_feats']
        att_masks = rela_data['att_masks']
        rela_feats = rela_data['rela_feats']
        rela_masks = rela_data['rela_masks']
        if self.use_attr_info == 1:
            attr_feats = rela_data['attr_feats']
            attr_masks = rela_data['attr_masks']

        att_feats_size = att_feats.size()
        N_att = att_feats_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = N_att / seq_per_img
        N_att_new_max = -1
        for img_id in range(int(N_img)):
            if self.use_attr_info != 0:
                N_att_new_max = \
                    max(N_att_new_max, torch.sum(rela_masks[img_id * seq_per_img, :]) +
                        torch.sum(att_masks[img_id * seq_per_img, :]) + torch.sum(attr_masks[img_id * seq_per_img, :, 0]))
            else:
                N_att_new_max = \
                    max(N_att_new_max, torch.sum(rela_masks[img_id * seq_per_img, :]) +
                        torch.sum(att_masks[img_id * seq_per_img, :]))
        att_masks_new = torch.zeros([N_att, int(N_att_new_max)]).cuda()
        att_feats_new = torch.zeros([N_att, int(N_att_new_max), self.rnn_size]).cuda()
        for img_id in range(int(N_img)):
            N_rela = int(torch.sum(rela_masks[img_id * seq_per_img, :]))
            N_box = int(torch.sum(att_masks[img_id * seq_per_img, :]))
            if self.use_attr_info == 1:
                N_attr = int(torch.sum(attr_masks[img_id * seq_per_img, :, 0]))
            else:
                N_attr = 0

            att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :] = \
                att_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box, :]
            if N_rela > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela, :] = \
                    rela_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_rela, :]
            if N_attr > 0:
                att_feats_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela: N_box + N_rela + N_attr, :] = \
                    attr_feats[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_attr, :]
            att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, 0:N_box] = 1
            if N_rela > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box:N_box + N_rela] = 1
            if N_attr > 0:
                att_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, N_box + N_rela:N_box + N_rela + N_attr] = 1

        return att_feats_new, att_masks_new

    def _extract_hs(self, fc_feats, att_feats, seq, att_masks=None, rela_data=None, ssg_data=None):
        """
        extract hidden states
        :param fc_feats:
        :param att_feats:
        :param seq:
        :param att_masks:
        :param rela_data:
        :param ssg_data:
        :return:
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.rnn_size)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new = self.sg_gfc(ssg_data, 'ssg')
            obj_feats, rela_feats, attr_feats = self.merge_ssg_att(ssg_data_new)

        for i in range(seq.size(1) - 1):
            it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_feats, obj_feats, rela_feats, attr_feats, state)
            outputs[:, i] = state[0][-1]

        return outputs

    def _extract_e(self, fc_feats, att_feats, seq, att_masks=None, rela_data=None, ssg_data=None):
        """
        extract embeddings
        :param fc_feats:
        :param att_feats:
        :param seq:
        :param att_masks:
        :param rela_data:
        :param ssg_data:
        :return:
        """
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.rnn_size)

        if self.use_ssg:
            fc_feats = self.fc_embed(fc_feats)
            ssg_data_new = self.sg_gfc(ssg_data, 'ssg')
            obj_feats, rela_feats, attr_feats = self.merge_ssg_att(ssg_data_new)

        return att_feats, att_masks

    def sg_gfc(self, sg_data, type):
        """
        use sentence scene graph's graph network to embed feats,
        :param sg_data: one dict which contains the following data:
               sg_data[type + '_rela_matrix']: relationship matrix for isg/ssg data,
                    [N_att, N_rela_max, 3] array
               sg_data[type + '_rela_masks']: relationship masks for isg/ssg data,
                    [N_att, N_rela_max]
               sg_data[type + '_obj']: obj index for isg/ssg data, [N_att, N_obj_max]
               sg_data[type + '_obj_masks']: obj masks, [N_att, N_obj_max]
               sg_data[type + '_attr']: attribute indexes, [N_att, N_obj_max, N_attr_max]
               sg_data[type + '_attr_masks']: attribute masks, [N_att, N_obj_max, N_attr_max]
        :return: sg_data_new one dict which contains the following data:
                 sg_data_new[type + '_rela_feats']: relationship embeddings, [N_att, N_rela_max, rnn_size]
                 sg_data_new[type + '_rela_masks']: equal to sg_data['ssg_rela_masks']
                 sg_data_new[type + '_obj_feats']: obj embeddings, [N_att, N_obj_max, rnn_size]
                 sg_data_new[type + '_obj_masks']: equal to sg_data[type + '_obj_masks']
                 sg_data_new[type + '_attr_feats']: attributes embeddings, [N_att, N_attr_max, rnn_size]
                 sg_data_new[type + '_attr_masks']: equal to sg_data[type + '_attr_masks']
        """
        sg_data_new = {}
        sg_data_new[type + '_rela_masks'] = sg_data[type + '_rela_masks']
        sg_data_new[type + '_obj_masks'] = sg_data[type + '_obj_masks']
        sg_data_new[type + '_attr_masks'] = sg_data[type + '_attr_masks']

        sg_obj = sg_data[type + '_obj']
        sg_obj_masks = sg_data[type + '_obj_masks']
        sg_attr = sg_data[type + '_attr']
        sg_attr_masks = sg_data[type + '_attr_masks']
        sg_rela_matrix = sg_data[type + '_rela_matrix']
        sg_rela_masks = sg_data[type + '_rela_masks']

        sg_obj_feats = torch.zeros([sg_obj.size()[0], sg_obj.size()[1], self.rnn_size]).cuda()
        sg_rela_feats = torch.zeros([sg_rela_matrix.size()[0], sg_rela_matrix.size()[1], self.rnn_size]).cuda()
        sg_attr_feats = torch.zeros([sg_attr.size()[0], sg_attr.size()[1], self.rnn_size]).cuda()
        sg_attr_masks_new = torch.zeros(sg_obj.size()).cuda()

        sg_obj_size = sg_obj.size()
        N_att = sg_obj_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att / seq_per_img)

        for img_id in range(N_img):
            N_obj = int(torch.sum(sg_obj_masks[img_id * seq_per_img, :]))
            if N_obj == 0:
                continue
            obj_feats_ori = self.embed(sg_obj[img_id * seq_per_img, :N_obj].cuda().long())
            obj_feats_temp = self.obj_obj_fc(obj_feats_ori)
            obj_num = np.ones([N_obj, ])

            N_rela = int(torch.sum(sg_rela_masks[img_id * seq_per_img, :]))
            rela_feats_temp = torch.zeros([N_rela, self.rnn_size])
            for rela_id in range(N_rela):
                sbj_id = int(sg_rela_matrix[img_id * seq_per_img, rela_id, 0])
                obj_id = int(sg_rela_matrix[img_id * seq_per_img, rela_id, 1])
                rela_index = sg_rela_matrix[img_id * seq_per_img, rela_id, 2]
                sbj_feat = obj_feats_ori[sbj_id]
                obj_feat = obj_feats_ori[obj_id]
                rela_feat = self.embed(rela_index.cuda().long())
                obj_feats_temp[sbj_id] = obj_feats_temp[sbj_id] + self.sbj_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
                obj_num[sbj_id] = obj_num[sbj_id] + 1.0
                obj_feats_temp[obj_id] = obj_feats_temp[obj_id] + self.obj_rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
                obj_num[obj_id] = obj_num[obj_id] + 1.0
                rela_feats_temp[rela_id] = self.rela_fc(torch.cat((sbj_feat, obj_feat, rela_feat)))
            for obj_id in range(N_obj):
                obj_feats_temp[obj_id] = obj_feats_temp[obj_id] / obj_num[obj_id]

            attr_feats_temp = torch.zeros([N_obj, self.rnn_size]).cuda()
            obj_attr_ids = 0
            for obj_id in range(N_obj):
                N_attr = int(torch.sum(sg_attr_masks[img_id * seq_per_img, obj_id, :]))
                if N_attr != 0:
                    attr_feat_ori = self.embed(sg_attr[img_id * seq_per_img, obj_id, :N_attr].cuda().long())
                    for attr_id in range(N_attr):
                        attr_feats_temp[obj_attr_ids] = attr_feats_temp[obj_attr_ids] + \
                                                        self.attr_fc(torch.cat((obj_feats_ori[obj_id], attr_feat_ori[attr_id])))
                    attr_feats_temp[obj_attr_ids] = attr_feats_temp[obj_attr_ids] / (N_attr + 0.0)
                    obj_attr_ids += 1
            N_obj_attr = obj_attr_ids
            sg_attr_masks_new[img_id * seq_per_img:(img_id + 1) * seq_per_img, :N_obj_attr] = 1

            sg_obj_feats[img_id * seq_per_img: (img_id + 1) * seq_per_img, :N_obj, :] = obj_feats_temp
            if N_rela != 0:
                sg_rela_feats[img_id * seq_per_img: (img_id + 1) * seq_per_img, :N_rela, :] = rela_feats_temp
            if N_obj_attr != 0:
                sg_attr_feats[img_id * seq_per_img: (img_id + 1) * seq_per_img, :N_obj_attr, :] = attr_feats_temp[:N_obj_attr]

        sg_data_new[type + '_obj_feats'] = sg_obj_feats
        sg_data_new[type + '_rela_feats'] = sg_rela_feats
        sg_data_new[type + '_attr_feats'] = sg_attr_feats
        sg_data_new[type + '_attr_masks'] = sg_attr_masks_new
        return sg_data_new

    def merge_sg_att(self, sg_data_new, type):
        """
        merge ssg_obj_feats, ssg_rela_feats, ssg_attr_feats together
        :param ssg_data_new:
        :return: att_feats: [N_att, N_att_max, rnn_size]
                 att_masks: [N_att, N_att_max]
        """
        sg_obj_feats = sg_data_new[type + '_obj_feats']
        sg_rela_feats = sg_data_new[type + '_rela_feats']
        sg_attr_feats = sg_data_new[type + '_attr_feats']
        sg_rela_masks = sg_data_new[type + '_rela_masks']
        sg_obj_masks = sg_data_new[type + '_obj_masks']
        sg_attr_masks = sg_data_new[type + '_attr_masks']

        ssg_obj_size = sg_obj_feats.size()
        N_att = ssg_obj_size[0]
        if self.index_eval == 1:
            seq_per_img = 1
        else:
            seq_per_img = self.seq_per_img
        N_img = int(N_att / seq_per_img)

        N_att_max = -1
        for img_id in range(N_img):
            N_rela = int(torch.sum(sg_rela_masks[img_id * seq_per_img, :]))
            N_obj = int(torch.sum(sg_obj_masks[img_id * seq_per_img, :]))
            N_attr = int(torch.sum(sg_attr_masks[img_id * seq_per_img, :]))

        if N_rela != 0:
            rela_feats = self.self_att_rel(sg_rela_feats)
        else:
            rela_feats = torch.zeros([sg_rela_feats.shape[0], self.rnn_size]).cuda()

        if N_obj != 0:
            obj_feats = self.self_att_obj(sg_obj_feats)
        else:
            obj_feats = torch.zeros([sg_obj_feats.shape[0], self.rnn_size]).cuda()

        if N_attr != 0:
            attr_feats = self.self_att_atr(sg_attr_feats)
        else:
            attr_feats = torch.zeros([sg_attr_feats.shape[0], self.rnn_size]).cuda()

        return obj_feats, rela_feats, attr_feats

    def _forward(self, real_A_obj, real_A_rel, real_A_atr, seq):
        batch_size = real_A_obj.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        fc_feats = torch.zeros([batch_size, self.rnn_size]).cuda()
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)

        obj_feats = real_A_obj
        rela_feats = real_A_rel
        attr_feats = real_A_atr

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, fc_feats, obj_feats, rela_feats, attr_feats, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, obj_feats, rela_feats, attr_feats, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, obj_feats, rela_feats, state, attr_feats)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, isg_data=None, ssg_data=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats = self.fc_embed(fc_feats)
        ssg_data_new = self.sg_gfc(ssg_data, 'ssg')
        _obj_feats, _rela_feats, _attr_feats = self.merge_sg_att(ssg_data_new, 'ssg')

        obj_feats = _obj_feats
        rela_feats = _rela_feats
        attr_feats = _attr_feats

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_obj_feats = obj_feats[k:k + 1].expand(*((beam_size,) + obj_feats.size()[1:])).contiguous()
            tmp_rela_feats = rela_feats[k:k + 1].expand(*((beam_size,) + rela_feats.size()[1:])).contiguous()
            tmp_attr_feats = attr_feats[k:k + 1].expand(*((beam_size,) + attr_feats.size()[1:])).contiguous() if attr_feats is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_obj_feats, tmp_rela_feats, tmp_attr_feats, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_obj_feats, tmp_rela_feats, tmp_attr_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, isg_data=None, ssg_data=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, isg_data, ssg_data, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        if self.use_isg:
            isg_obj_feats, isg_rela_feats, isg_attr_feats = self.merge_sg_att(self.sg_gfc(isg_data, 'isg'), 'isg')

        fc_feats = self.fc_embed(fc_feats)
        ssg_data_new = self.sg_gfc(ssg_data, 'ssg')
        _obj_feats, _rela_feats, _attr_feats = self.merge_sg_att(ssg_data_new, 'ssg')

        attr_feats = _obj_feats
        rela_feats = _obj_feats
        obj_feats = _obj_feats

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, fc_feats, obj_feats, rela_feats, attr_feats, state)

        return seq, seqLogprobs

    def _sample_gan(self, real_A_obj, real_A_rel, real_A_atr, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam_gan(real_A_obj, real_A_rel, real_A_atr, opt)

        batch_size = real_A_obj.size(0)
        state = self.init_hidden(batch_size)

        fc_feats = torch.zeros([batch_size, self.rnn_size]).cuda()

        obj_feats = real_A_obj
        rela_feats = real_A_rel
        attr_feats = real_A_atr

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, fc_feats, obj_feats, rela_feats, attr_feats, state)

        return seq, seqLogprobs

    def _sample_beam_gan(self, real_A_obj, real_A_rel, real_A_atr, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = real_A_obj.size(0)

        fc_feats = torch.zeros([batch_size, self.rnn_size]).cuda()
        obj_feats = real_A_obj
        rela_feats = real_A_rel
        attr_feats = real_A_atr

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_obj_feats = obj_feats[k:k + 1].expand(*((beam_size,) + obj_feats.size()[1:])).contiguous()
            tmp_rela_feats = rela_feats[k:k + 1].expand(*((beam_size,) + rela_feats.size()[1:])).contiguous()
            tmp_attr_feats = attr_feats[k:k + 1].expand(*((beam_size,) + attr_feats.size()[1:])).contiguous() if attr_feats is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_obj_feats, tmp_rela_feats, tmp_attr_feats, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_obj_feats, tmp_rela_feats, tmp_attr_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), [obj_feats, rela_feats, attr_feats], [None, None, None]

    def _sample_i2t(self, netG_obj, netG_rel, netG_atr, fc_feats, isg_data=None, ssg_data=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam_i2t(netG_obj, netG_rel, netG_atr, fc_feats, isg_data, ssg_data, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        fc_feats = self.fc_embed(fc_feats)
        ssg_data_new = self.sg_gfc(ssg_data, 'ssg')
        _obj_feats, _rela_feats, _attr_feats = self.merge_sg_att(ssg_data_new, 'ssg')

        if netG_obj is not None:
            obj_feats = netG_obj(_obj_feats)
        else:
            obj_feats = _obj_feats

        if netG_rel is not None:
            rela_feats = netG_rel(_rela_feats)
        else:
            rela_feats = _rela_feats

        if netG_atr is not None:
            attr_feats = netG_atr(_attr_feats)
        else:
            attr_feats = _attr_feats

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:, t - 1] = it
                seqLogprobs[:, t - 1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, fc_feats, obj_feats, rela_feats, attr_feats, state)

        return seq, seqLogprobs, None, None

    def _sample_beam_i2t(self, netG_obj, netG_rel, netG_atr, fc_feats, isg_data=None, ssg_data=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats = self.fc_embed(fc_feats)
        ssg_data_new = self.sg_gfc(ssg_data, 'ssg')
        _obj_feats, _rela_feats, _attr_feats = self.merge_sg_att(ssg_data_new, 'ssg')

        if netG_obj is not None:
            obj_feats = netG_obj(_obj_feats)
        else:
            obj_feats = _obj_feats

        if netG_rel is not None:
            rela_feats = netG_rel(_rela_feats)
        else:
            rela_feats = _rela_feats

        if netG_atr is not None:
            attr_feats = netG_atr(_attr_feats)
        else:
            attr_feats = _attr_feats

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_obj_feats = obj_feats[k:k + 1].expand(*((beam_size,) + obj_feats.size()[1:])).contiguous()
            tmp_rela_feats = rela_feats[k:k + 1].expand(*((beam_size,) + rela_feats.size()[1:])).contiguous()
            tmp_attr_feats = attr_feats[k:k + 1].expand(*((beam_size,) + attr_feats.size()[1:])).contiguous() if attr_feats is not None else None

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_obj_feats, tmp_rela_feats, tmp_attr_feats, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_obj_feats, tmp_rela_feats, tmp_attr_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), [obj_feats, rela_feats, attr_feats], [None, None, None]

class GTssgCore_UP_SepSelfAtt_SEP(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(GTssgCore_UP_SepSelfAtt_SEP, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.mapping_triplet = nn.Sequential(nn.Linear(opt.rnn_size * 3, opt.rnn_size),
                                         nn.ReLU(inplace=True))
                                             # , nn.Dropout(opt.drop_prob_lm))
        self.lstm1 = nn.LSTMCell(opt.input_encoding_size, opt.rnn_size)  # we, h^2_t-1
        self.lstm2 = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v

    def forward(self, xt, fc_feats, obj_feats, rela_feats, state, attr_feats=None):
        prev_h = state[0][-1]
        # lstm1_input = torch.cat([prev_h, xt], 1)
        lstm1_input = torch.cat([xt], 1)

        # state[0][0] means the hidden state c in first lstm
        # state[1][0] means the cell h in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state
        h_att, c_att = self.lstm1(lstm1_input, (state[0][0], state[1][0]))

        att = self.mapping_triplet(torch.cat([obj_feats, rela_feats, attr_feats], 1))
        # att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lstm2_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lstm2(lstm2_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

from .FCModel import LSTMCore

class GTssgModel_UP_SepSelfAtt_SEP(AttModel):
    def __init__(self, opt):
        super(GTssgModel_UP_SepSelfAtt_SEP, self).__init__(opt)
        self.num_layers = 2
        self.core = GTssgCore_UP_SepSelfAtt_SEP(opt)

class SelfAttention(nn.Module):
    def __init__(self, opt):
        super(SelfAttention, self).__init__()
        self.hidden_dim = opt.rnn_size
        self.projection = nn.Sequential(
            nn.Linear(opt.rnn_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # outputs = SpectralNorm(outputs)
        return outputs

class Log_Rbm(nn.Module):
    def __init__(self, D_in, R, D_out):
        super(Log_Rbm, self).__init__()
        self.D_in = D_in
        self.R = R
        self.D_out = D_out
        self.w = nn.Linear(D_in, R, bias=False)
        u_init = np.random.rand(self.R, D_out) / 1000
        u_init = np.float32(u_init)
        self.u = torch.from_numpy(u_init).cuda().requires_grad_()

    def forward(self, x):
        v = self.w(x)  # x Batch*D_in, v Batch*R

        v = v.unsqueeze(2).expand(-1, -1, self.D_out)  # v: Batch*R*D_out
        u = self.u.unsqueeze(0).expand(v.size(0), -1, -1)  # u: Batch*R*D_out
        v = v + u
        v = torch.exp(v)
        v = v + 1
        y = torch.log(v)
        y = torch.sum(v, 1)
        return y