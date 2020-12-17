# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
from gan_utils import *
import opts
import models
from dataloader_up_disorder import *
from dataloader_up_mt import *
from dataloaderraw import *
import eval_utils_cn_copy3 as eval_utils
import argparse
import misc.utils as utils
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--caption_model', type=str, default="gtssg_sep_self_att_sep_v2", help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, gtssg, gtssg_up')
parser.add_argument('--model', type=str, default='unpaired_image_caption_revise/save/20200317_154541_gtssg_sep_self_att_sep_v2/model.pth', help='path to model to evaluate')
parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                    'infos.pkl'         : configuration;
                    'checkpoint'        : paths to model file(s) (created by tf).
                                          Note: this file contains absolute paths, be careful when moving files around;
                    'model.ckpt-*'      : file(s) with model definition (created by tf)
                """)
parser.add_argument('--cnn_model', type=str, default='resnet101', help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='unpaired_image_caption_revise/save/20200317_154541_gtssg_sep_self_att_sep_v2/infos.pkl', help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0, help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=100, help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0, help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1, help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=1, help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1, help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0, help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=5, help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
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
parser.add_argument('--input_label_h5', type=str, default='data/aic_process/cocobu_ALL_11683_v2_COCOCN_t5_isg_label.h5', help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='data/aic_process/cocobu_ALL_11683_v2_COCOCN_t5_isg.json', help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
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
parser.add_argument('--use_spectral_norm', type=int, default=0, help='If debug')
parser.add_argument('--use_isg', type=int, default=0, help='If use ssg')
parser.add_argument('--use_ssg', type=int, default=1, help='If use ssg')
parser.add_argument('--ssg_dict_path', type=str, default='data/aic_process/ALL_11683_v2_COCOCN_spice_sg_dict_t5.npz_revise.npz', help='path to the sentence scene graph directory')
parser.add_argument('--gru_t', type=int, default=4, help='the numbers of gru will iterate')
parser.add_argument('--index_eval', type=int, default=1, help='whether eval or not')
parser.add_argument('--input_rela_dir', type=str, default='data/cocotalk_rela', help='path to the directory containing the relationships of att feats')
parser.add_argument('--input_ssg_dir', type=str, default='data/coco_cn/coco_COCOCN_pred_fuse_sg_v2_small', help='path to the directory containing the ground truth sentence scene graph')
# parser.add_argument('--input_ssg_dir', type=str, default='data/aic_process/ALL_11683_v2_COCOCN_spice_sg_dict_t5', help='path to the directory containing the ground truth sentence scene graph')
parser.add_argument('--output_cap_path', type=str, default='0', help='file which save the result')
opt = parser.parse_args()

# # parameters need to be edit
opt.input_json='data/coco_cn/cocobu_gan_isg.json'
opt.input_label_h5='data/coco_cn/cocobu_gan_isg_label.h5'
opt.ssg_dict_path='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'
opt.input_ssg_dir='data/coco_cn/coco_COCOCN_pred_fuse_sg_v3_newid_small_en_and_trans_ssg_en'


opt.caption_model='gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate'
opt.start_from='save_for_finetune/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/'
opt.model= '/GPUFS/hku_sac_yu_vs/unpaired_image_caption_edit_local_cloud/unpaired_image_caption_revise/save_for_finetune/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/model-best.pth'
opt.infos_path='/GPUFS/hku_sac_yu_vs/unpaired_image_caption_edit_local_cloud/unpaired_image_caption_revise/save_for_finetune/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/infos-best.pkl'

opt.num_images=-1
opt.gpu=0

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    # opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id

ignore = ['num_images', "use_gfc", "use_isg", "ssg_dict_path", "input_json", "input_label_h5", "id", "batch_size", "start_from", "language_eval", "use_rela", "input_ssg_dir", "ssg_dict_path", "input_rela_dir", "use_spectral_norm", "beam_size",'gpu','caption_model']
beam_size = opt.beam_size
for k in vars(infos['opt']).keys():
    if k != 'model':
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping
opt.vocab_size=len(vocab)


# # Setup the model
# try:
#     opt.caption_model=opt.caption_model_zh
# except:
#     opt.caption_model=opt.caption_model
model = models.setup(opt)
print ('load parameters from {}'.format(opt.model))
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

if opt.use_isg:
    if opt.start_from is not None:
        ckpt_G = torch.load(os.path.join(opt.checkpoint_path, 'model_G-best.pth'))
        ckpt_D = torch.load(os.path.join(opt.checkpoint_path, 'model_D-best.pth'))
    if opt.gan_type == 0:
        netD_A_obj = GAN_init_D(opt, Discriminator(opt), type='netD_A_obj').cuda().eval()
        netD_A_rel = GAN_init_D(opt, Discriminator(opt), type='netD_A_rel').cuda().eval()
        netD_A_atr = GAN_init_D(opt, Discriminator(opt), type='netD_A_atr').cuda().eval()

        netD_B_obj = GAN_init_D(opt, Discriminator(opt), type='netD_B_obj').cuda().eval()
        netD_B_rel = GAN_init_D(opt, Discriminator(opt), type='netD_B_rel').cuda().eval()
        netD_B_atr = GAN_init_D(opt, Discriminator(opt), type='netD_B_atr').cuda().eval()

        netG_A_obj = GAN_init_G(opt, Generator(opt), type='netG_A_obj').cuda().eval()
        netG_A_rel = GAN_init_G(opt, Generator(opt), type='netG_A_rel').cuda().eval()
        netG_A_atr = GAN_init_G(opt, Generator(opt), type='netG_A_atr').cuda().eval()

        netG_B_obj = GAN_init_G(opt, Generator(opt), type='netG_B_obj').cuda().eval()
        netG_B_rel = GAN_init_G(opt, Generator(opt), type='netG_B_rel').cuda().eval()
        netG_B_atr = GAN_init_G(opt, Generator(opt), type='netG_B_atr').cuda().eval()
    elif opt.gan_type == 1:
        netD_A_obj = GAN_init_D(opt, Discriminator(opt), type='netD_A_obj').cuda().eval()
        netD_A_rel = GAN_init_D(opt, Discriminator(opt), type='netD_A_rel').cuda().eval()
        netD_A_atr = None

        netD_B_obj = GAN_init_D(opt, Discriminator(opt), type='netD_B_obj').cuda().eval()
        netD_B_rel = GAN_init_D(opt, Discriminator(opt), type='netD_B_rel').cuda().eval()
        netD_B_atr = None

        netG_A_obj = GAN_init_G(opt, Generator(opt), type='netG_A_obj').cuda().eval()
        netG_A_rel = GAN_init_G(opt, Generator(opt), type='netG_A_rel').cuda().eval()
        netG_A_atr = None

        netG_B_obj = GAN_init_G(opt, Generator(opt), type='netG_B_obj').cuda().eval()
        netG_B_rel = GAN_init_G(opt, Generator(opt), type='netG_B_rel').cuda().eval()
        netG_B_atr = None
    elif opt.gan_type == 2:
        netD_A_obj = GAN_init_D(opt, Discriminator(opt), type='netD_A_obj').cuda().train()
        netD_A_rel = None
        netD_A_atr = None

        netD_B_obj = GAN_init_D(opt, Discriminator(opt), type='netD_B_obj').cuda().train()
        netD_B_rel = None
        netD_B_atr = None

        netG_A_obj = GAN_init_G(opt, Generator(opt), type='netG_A_obj').cuda().train()
        netG_A_rel = None
        netG_A_atr = None

        netG_B_obj = GAN_init_G(opt, Generator(opt), type='netG_B_obj').cuda().train()
        netG_B_rel = None
        netG_B_atr = None

crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader_UP(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 'coco_json': opt.coco_json, 'batch_size': opt.batch_size, 'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
use_mapping = False
if getattr(opt, 'use_isg', 0) == 1:
    loss, split_predictions, lang_stats = eval_utils.eval_split_up(model, use_mapping, None, None, None, crit, loader, vars(opt))
else:
    loss, split_predictions, lang_stats = eval_utils.eval_split_nmt(model, crit, loader, vars(opt))
    # loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

text_file = open('res_' + opt.id + '.txt', "aw")
text_file.write('{0}\n'.format(opt.model))
text_file.write('beam_size {0}\n'.format(beam_size))
text_file.write('{0}\n'.format(lang_stats))
text_file.close()

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
