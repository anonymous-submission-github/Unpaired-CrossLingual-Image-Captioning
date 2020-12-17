from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as n

import time
import os
from six.moves import cPickle
import pdb
import opts
import models
from dataloader_up_ft import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
from eval_utils import *
import torch


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--caption_model', type=str, default="up_gtssg_sep_self_att_sep_extract", help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, gtssg, gtssg_up')
parser.add_argument('--model', type=str, default='/home/jxgu/github/unparied_im2text_graph/save/20190201_113648_gtssg_sep_self_att_sep/model-best.pth', help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101', help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='/home/jxgu/github/unparied_im2text_graph/save/20190201_113648_gtssg_sep_self_att_sep/infos-best.pkl', help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=50, help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=5000, help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0, help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1, help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0, help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1, help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0, help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=1, help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1, help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5, help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0, help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='', help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='', help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='', help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='', help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='', help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=0, help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, help='if we need to calculate loss.')
parser.add_argument('--verbose', type=int, default=0, help='if we need to print out all beam search beams.')
parser.add_argument('--use_rela', type=int, default=0, help='whether to use relationship matrix.')
parser.add_argument('--use_gru', type=int, default=0, help='whether to use relationship matrix.')
parser.add_argument('--use_gfc', type=int, default=1, help='whether to use relationship matrix.')
parser.add_argument('--use_ssg', type=int, default=1, help='If use ssg')
parser.add_argument('--ssg_dict_path', type=str, default='data/coco_is_dict.npy', help='path to the sentence scene graph directory')
parser.add_argument('--gru_t', type=int, default=4, help='the numbers of gru will iterate')
parser.add_argument('--index_eval', type=int, default=1, help='whether eval or not')
parser.add_argument('--input_rela_dir', type=str, default='data/cocotalk_rela', help='path to the directory containing the relationships of att feats')
parser.add_argument('--input_ssg_dir', type=str, default='data/coco_spice_sg', help='path to the directory containing the ground truth sentence scene graph')
parser.add_argument('--output_cap_path', type=str, default='0', help='file which save the result')
parser.add_argument('--gpu', type=int, default='1', help='gpu_id')
opt = parser.parse_args()


opt.caption_model='up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2_extract'
opt.start_from='unpaired_image_caption_revise/save_final_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/'
opt.checkpoint_path='unpaired_image_caption_revise/save_final_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/'
opt.model='unpaired_image_caption_revise/save_final_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/model-best.pth'
opt.infos_path='unpaired_image_caption_revise/save_final_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/infos-best.pkl'

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)


# store cococn ssg for training gan
opt.input_json_isg='data/coco_cn/cocobu_gan_isg.json'
opt.input_json='data/coco_cn/cocobu_gan_ssg.json'
opt.input_isg_dir='data/coco_cn/coco_COCOCN_pred_fuse_sg_v3_newid_small_en'
opt.input_json='data/coco_cn/cocobu_gan_isg.json'
opt.input_ssg_dir='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'
opt.ssg_dict_path='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'
opt.gpu=0
opt.split='test'


os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
    opt.input_label_h5 = 'data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_testcococn1000_label.h5'
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json

opt.batch_size = 50
if len(opt.id) == 0:
    opt.id = infos['opt'].id

ignore = ['input_label_h5','input_json',"caption_model", "beam_size", "use_gfc", "id", "batch_size", "start_from","checkpoint_path", "language_eval", "use_rela", "input_ssg_dir","input_isg_dir", "ssg_dict_path", "input_rela_dir",'gpu']
beam_size = opt.beam_size
for k in vars(infos['opt']).keys():
    if k != 'model':
        if k not in ignore:
            if k in vars(opt):
                print (vars(opt)[k])
                print (vars(infos['opt'])[k])
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

opt.use_isg = 1

vocab = infos['vocab']  # ix -> word mapping
opt.vocab_size=len(vocab)
opt.freeze_i2t=1
# ssg_dict_info = np.load(opt.ssg_dict_path, allow_pickle=True)['spice_dict'][()]
# vocab = ssg_dict_info['ix_to_word']

# Setup the model
model = models.setup(opt)
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()
split = opt.split


# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader_UP_FT(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 'coco_json': opt.coco_json, 'batch_size': opt.batch_size, 'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

rela_dict_path = 'data/rela_dict.npy'
rela_dict_info = np.load(rela_dict_path,allow_pickle=True)
rela_dict_info = rela_dict_info[()]

loader.reset_iterator(split)

n = 0
predictions = []
while True:
    data = loader.get_batch(split)
    n = n + loader.batch_size
    print("Ix={}/{}".format(n, split))
    # forward the model to also get generated samples for each image
    # Only leave one feature for each image, in case duplicate sample
    tmp = [data['fc_feats'], data['labels'], data['masks'], data['att_feats'], data['att_masks'],
           data['isg_rela_matrix'], data['isg_rela_masks'], data['isg_obj'], data['isg_obj_masks'], data['isg_attr'], data['isg_attr_masks'],
           data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'], data['ssg_obj_masks'], data['ssg_attr'], data['ssg_attr_masks']]
    try:
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    except:
        pdb.set_trace()
    fc_feats, labels, masks, att_feats, att_masks, \
    isg_rela_matrix, isg_rela_masks, isg_obj, isg_obj_masks, isg_attr, isg_attr_masks, \
    ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks = tmp

    # image graph domain
    isg_data = {}
    isg_data['att_feats'] = att_feats
    isg_data['att_masks'] = att_masks

    isg_data['isg_rela_matrix'] = isg_rela_matrix
    isg_data['isg_rela_masks'] = isg_rela_masks
    isg_data['isg_obj'] = isg_obj
    isg_data['isg_obj_masks'] = isg_obj_masks
    isg_data['isg_attr'] = isg_attr
    isg_data['isg_attr_masks'] = isg_attr_masks
    # text graph domain
    ssg_data = {}
    ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
    ssg_data['ssg_rela_masks'] = ssg_rela_masks
    ssg_data['ssg_obj'] = ssg_obj
    ssg_data['ssg_obj_masks'] = ssg_obj_masks
    ssg_data['ssg_attr'] = ssg_attr
    ssg_data['ssg_attr_masks'] = ssg_attr_masks

    # forward the model to also get generated samples for each image
    with torch.no_grad():
        gen_outputs, gen_ssg, gen_isg, gen_ssg_before_att_, gen_isg_before_att_  = model(fc_feats, labels, isg_data, ssg_data)

    for i in range(len(data['path'])):
        """
        gen_isg_before_att = {}
        gen_isg_before_att['isg_rela_feats'] = gen_ssg_before_att_['isg_rela_feats'][i * 5, :, :].data.cpu().numpy()
        gen_isg_before_att['isg_obj_feats'] = gen_ssg_before_att_['isg_obj_feats'][i * 5, :, :].data.cpu().numpy()
        gen_isg_before_att['isg_attr_feats'] = gen_ssg_before_att_['isg_attr_feats'][i * 5, :, :].data.cpu().numpy()

        gen_ssg_before_att = {}
        gen_ssg_before_att['ssg_rela_feats'] = gen_isg_before_att_['ssg_rela_feats'][i * 5, :, :].data.cpu().numpy()
        gen_ssg_before_att['ssg_obj_feats'] = gen_isg_before_att_['ssg_obj_feats'][i * 5, :, :].data.cpu().numpy()
        gen_ssg_before_att['ssg_attr_feats'] = gen_isg_before_att_['ssg_attr_feats'][i * 5, :, :].data.cpu().numpy()

        np.save('/media/jxgu/work/datasets/mscoco/coco_graph_extract_ft_ssg_before_att/' + str(data['path'][i]) + '.npy', gen_isg_before_att)
        np.save('/media/jxgu/work/datasets/mscoco/coco_graph_extract_ft_isg_before_att/' + str(data['path'][i]) + '.npy', gen_ssg_before_att)
        """


        # # rcsls+joint +subpmap+global training feature extraction
        save_ssg_path='data/coco_graph_extract_ft_ssg_joint_rcsls_submap_global_naacl_self_gate/'
        save_isg_path='data/coco_graph_extract_ft_isg_joint_rcsls_submap_global_naacl_self_gate/'
        if not os.path.exists(save_ssg_path): os.makedirs(save_ssg_path)
        if not os.path.exists(save_isg_path): os.makedirs(save_isg_path)
        np.save(save_ssg_path + str(data['path'][i]) + '.npy' , torch.stack(gen_ssg)[:, i , :].data.cpu().numpy())
        np.save(save_isg_path + str(data['path_isg'][i]) + '.npy', torch.stack(gen_isg)[:, i, :].data.cpu().numpy())

    # if we wrapped around the split or used up val imgs budget then bail
    if data['bounds']['wrapped']:
        break