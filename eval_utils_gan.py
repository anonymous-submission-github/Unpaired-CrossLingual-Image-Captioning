from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval(dataset, preds, model_id, split):
    import sys
    # sys.path.append("coco-caption")
    # annFile = 'coco-caption/annotations/captions_val2014.json'
    # from pycocotools.coco import COCO
    # from pycocoevalcap.eval import COCOEvalCap

    sys.path.append("coco-cn-cap-eval")
    from evals.coco_caption.pycxtools.coco import COCO
    from evals.coco_caption.pycxevalcap.eval import COCOEvalCap
    # annFile = 'data/coco_cn/COCOCN_val_test_isg.json'  # 1 gt sentence for testing
    annFile = 'data/coco_cn/COCOCN_val_test_isg_v2.json' # 6 gt sentences for testing


    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    cache_path_1 = os.path.join('eval_results/', model_id + '_' + split +'_scores'+'.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...
    print("Dump to json file {}".format(cache_path))
    # with io.open(cache_path+'plus', 'w', encoding="utf-8") as outfile:
    #     outfile.write(json.dumps(preds_filt, ensure_ascii=False))
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path_1, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    print("Dump to json file {}".format(cache_path_1))

    return out, imgToEval, cache_path

def eval_split_gan(opt, model, netG_obj, netG_rel, netG_atr, loader, loader_i2t, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = 5000
    criterionCycle = torch.nn.L1Loss()

    netG_rel.eval()
    netG_obj.eval()
    netG_atr.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    while True:
        data_gan = loader.get_batch(split)

        n = n + loader.batch_size

        tmp_gan = [data_gan['isg_feats'][:, 0, :], data_gan['isg_feats'][:, 1, :], data_gan['isg_feats'][:, 2, :],
                   data_gan['ssg_feats'][:, 0, :], data_gan['ssg_feats'][:, 1, :], data_gan['ssg_feats'][:, 2, :]]
        tmp_gan = [_ if _ is None else torch.from_numpy(_).float().cuda() for _ in tmp_gan]
        real_A_obj, real_A_rel, real_A_atr, real_B_obj, real_B_rel, real_B_atr = tmp_gan

        fake_B_obj = netG_obj(real_A_obj)
        fake_B_rel = netG_rel(real_A_rel)
        fake_B_atr = netG_atr(real_A_atr)

        loss = criterionCycle(real_A_obj, fake_B_obj.detach()) + criterionCycle(fake_B_rel, real_A_rel.detach()) + criterionCycle(fake_B_atr, real_A_atr.detach())
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_gan['bounds']['it_pos_now']
        ix1 = data_gan['bounds']['it_max']

        ix1 = min(ix1, num_images)
        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data_gan['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    if netG_rel is not None: netG_rel.train()
    if netG_obj is not None: netG_obj.train()
    if netG_atr is not None: netG_atr.train()

    return loss_sum/loss_evals

def eval_split_gan_v1(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, netG_B_obj, netG_B_rel, netG_B_atr, loader, loader_i2t, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = 5000
    criterionCycle = torch.nn.L1Loss()

    netG_A_obj.eval()
    netG_A_rel.eval()
    netG_A_atr.eval()

    netG_B_obj.eval()
    netG_B_rel.eval()
    netG_B_atr.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        start_time = time.time()
        data_gan = loader.get_batch(split)

        n = n + loader.batch_size

        tmp_gan = [data_gan['isg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp_gan = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp_gan]
        real_A_obj, real_A_rel, real_A_atr, real_B_obj, real_B_rel, real_B_atr, labels = tmp_gan

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            model.use_isg = 0
            fake_B_obj = netG_A_obj(real_A_obj)
            fake_B_rel = netG_A_rel(real_A_rel)
            fake_B_atr = netG_A_atr(real_A_atr)

            seq_isg = model(fake_B_obj, fake_B_rel, fake_B_atr, opt=eval_kwargs, mode='sample_gan')[0].data

        sents_isg = utils.decode_sequence(loader.get_vocab(), seq_isg, use_ssg=True)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:, 1:], use_ssg=True)
        for k, sent in enumerate(sents_isg):
            entry = {'image_id': data_gan['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data_gan['infos'][k]['file_path']
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_gan['bounds']['it_pos_now']
        ix1 = data_gan['bounds']['it_max']

        ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, time.time() - start_time))

        if data_gan['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    lang_stats, scores_each, cache_path = language_eval('coco', predictions, eval_kwargs['id'], 'test')

    return loss_sum/loss_evals, cache_path

def eval_split_gan_v2(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, netG_B_obj, netG_B_rel, netG_B_atr, loader, loader_i2t, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = 5000
    criterionCycle = torch.nn.L1Loss()

    netG_A_obj.eval()
    netG_A_rel.eval()
    netG_A_atr.eval()

    netG_B_obj.eval()
    netG_B_rel.eval()
    netG_B_atr.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        start_time = time.time()
        data_gan = loader.get_batch(split)

        n = n + loader.batch_size

        tmp_gan = [data_gan['isg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp_gan = [_ if _ is None else torch.from_numpy(_).float().cuda() for _ in tmp_gan]
        real_A_obj, real_A_rel, real_A_atr, real_B_obj, real_B_rel, real_B_atr, labels = tmp_gan

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            model.use_isg = 0
            combine = 0
            if combine == 0:
                fake_B_obj = netG_A_obj(real_A_obj)
                fake_B_rel = netG_A_rel(real_A_rel)
                fake_B_atr = netG_A_atr(real_A_atr)
            elif combine == 1:
                fake_B_obj = netG_B_obj(real_A_obj)
                fake_B_rel = netG_B_rel(real_A_rel)
                fake_B_atr = netG_B_atr(real_A_atr)
            elif combine == 2:
                fake_B_obj = netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj)
                fake_B_rel = netG_A_rel(real_A_rel) + netG_B_rel(real_A_rel)
                fake_B_atr = netG_A_atr(real_A_atr) + netG_B_atr(real_A_atr)
            elif combine == 3:
                fake_B_obj = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))/2
                fake_B_rel = (netG_A_rel(real_A_rel) + netG_B_rel(real_A_rel))/2
                fake_B_atr = (netG_A_atr(real_A_atr) + netG_B_atr(real_A_atr))/2
            elif combine == 4:
                fake_B_obj = netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj)
                fake_B_rel = netG_A_rel(real_A_rel) + netG_B_obj(real_A_obj)
                fake_B_atr = netG_A_atr(real_A_atr) + netG_B_obj(real_A_obj)
            elif combine == 5:
                fake_B_obj = netG_A_obj(real_A_obj) + netG_B_rel(real_A_rel)
                fake_B_rel = netG_A_rel(real_A_rel) + netG_B_rel(real_A_rel)
                fake_B_atr = netG_A_atr(real_A_atr) + netG_B_rel(real_A_rel)
            elif combine == 6:
                fake_B_obj = netG_A_obj(real_A_obj) + netG_B_atr(real_A_atr)
                fake_B_rel = netG_A_rel(real_A_rel) + netG_B_atr(real_A_atr)
                fake_B_atr = netG_A_atr(real_A_atr) + netG_B_atr(real_A_atr)
            elif combine == 7:
                fake_B_obj = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))/2
                fake_B_rel = (netG_A_rel(real_A_rel) + netG_B_obj(real_A_obj))/2
                fake_B_atr = (netG_A_atr(real_A_atr) + netG_B_obj(real_A_obj))/2
            elif combine == 8:
                fake_B_obj = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))/2
                fake_B_rel = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))/2
                fake_B_atr = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))/2
            elif combine == 9:
                fake_B_obj = (netG_A_rel(real_A_rel) + netG_B_rel(real_A_rel))/2
                fake_B_rel = (netG_A_rel(real_A_rel) + netG_B_rel(real_A_rel))/2
                fake_B_atr = (netG_A_rel(real_A_rel) + netG_B_rel(real_A_rel))/2
            elif combine == 10:
                fake_B_obj = (netG_A_atr(real_A_atr) + netG_B_atr(real_A_atr))/2
                fake_B_rel = (netG_A_atr(real_A_atr) + netG_B_atr(real_A_atr))/2
                fake_B_atr = (netG_A_atr(real_A_atr) + netG_B_atr(real_A_atr))/2
            elif combine == 11:
                fake_B_obj = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))
                fake_B_rel = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))
                fake_B_atr = (netG_A_obj(real_A_obj) + netG_B_obj(real_A_obj))
            elif combine == 12:
                fake_B_obj = real_A_obj + real_A_obj
                fake_B_rel = real_A_obj + real_A_obj
                fake_B_atr = real_A_obj + real_A_obj
            elif combine == 13:
                fake_B_obj = real_A_obj
                fake_B_rel = real_A_obj
                fake_B_atr = real_A_obj
            elif combine == 14:
                fake_B_obj = netG_A_obj(real_A_obj)
                fake_B_rel = netG_A_obj(real_A_obj)
                fake_B_atr = netG_A_obj(real_A_obj)
            elif combine == 15:
                fake_B_obj = real_A_obj
                fake_B_rel = real_A_rel
                fake_B_atr = real_A_atr
            elif combine == 16:
                fake_B_obj = real_B_obj
                fake_B_rel = real_B_rel
                fake_B_atr = real_B_atr

            seq_isg = model(fake_B_obj, fake_B_rel, fake_B_atr, opt=eval_kwargs, mode='sample_gan')[0].data

        sents_isg = utils.decode_sequence(loader.get_vocab(), seq_isg, use_ssg=True)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:, 1:], use_ssg=True)
        for k, sent in enumerate(sents_isg):
            entry = {'image_id': data_gan['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data_gan['infos'][k]['file_path']
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_gan['bounds']['it_pos_now']
        ix1 = data_gan['bounds']['it_max']

        ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, time.time() - start_time))

        if data_gan['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    lang_stats, scores_each, cache_path = language_eval('coco', predictions, eval_kwargs['id'], 'test')

    return loss_sum/loss_evals, cache_path

def eval_split_gan_v3(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, loader, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = 5000
    criterionCycle = torch.nn.L1Loss()

    netG_A_obj.eval()
    netG_A_rel.eval()
    netG_A_atr.eval()


    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        start_time = time.time()
        data_gan = loader.get_batch(split)

        n = n + loader.batch_size

        tmp_gan = [data_gan['isg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['ssg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp_gan = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp_gan]
        real_A_obj, real_A_rel, real_A_atr, real_B_obj, real_B_rel, real_B_atr, labels = tmp_gan

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            model.use_isg = 0
            combine = 0
            if combine == 0:
                fake_B_obj = real_A_obj
                fake_B_rel = netG_A_rel(real_A_rel)
                fake_B_atr = netG_A_atr(real_A_atr)
            elif combine == 1:
                fake_B_obj = real_A_obj
                fake_B_rel = real_A_rel
                fake_B_atr = real_A_atr
            elif combine == 2:
                fake_B_obj = real_B_obj
                fake_B_rel = real_B_rel
                fake_B_atr = real_B_atr

            seq_isg = model(fake_B_obj, fake_B_rel, fake_B_atr, opt=eval_kwargs, mode='sample_gan')[0].data

        sents_isg = utils.decode_sequence(loader.get_vocab(), seq_isg, use_ssg=True)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:, 1:], use_ssg=True)
        for k, sent in enumerate(sents_isg):
            entry = {'image_id': data_gan['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data_gan['infos'][k]['file_path']
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_gan['bounds']['it_pos_now']
        ix1 = data_gan['bounds']['it_max']

        ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, time.time() - start_time))

        if data_gan['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    lang_stats, scores_each, cache_path = language_eval('coco', predictions, eval_kwargs['id'], 'test')

    return lang_stats


def eval_split_g2t(opt, model, netG_obj, netG_rel, netG_atr, loader, loader_i2t, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = opt.num_images
    criterionCycle = torch.nn.L1Loss()

    if netG_rel is not None: netG_rel.eval()
    if netG_obj is not None: netG_obj.eval()
    if netG_atr is not None: netG_atr.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        start_time = time.time()
        data_gan = loader.get_batch(split)

        n = n + loader.batch_size

        tmp_gan = [data_gan['isg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp_gan = [_ if _ is None else torch.from_numpy(_).float().cuda() for _ in tmp_gan]
        real_A_obj, real_A_rel, real_A_atr, labels = tmp_gan

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            model.use_isg = 0
            fake_A_obj = netG_obj(real_A_obj)
            fake_A_rel = netG_rel(real_A_rel)
            fake_A_atr = netG_atr(real_A_atr)
            # fake_A_obj = real_A_obj
            # fake_A_rel = real_A_rel
            # fake_A_atr = real_A_atr

            seq_isg = model(fake_A_obj, fake_A_rel, fake_A_atr, opt=eval_kwargs, mode='sample_gan')[0].data

        sents_isg = utils.decode_sequence(loader.get_vocab(), seq_isg, use_ssg=True)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:, 1:], use_ssg=True)
        for k, sent in enumerate(sents_isg):
            entry = {'image_id': data_gan['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 1) == 1:
                entry['file_name'] = data_gan['infos'][k]['file_path']
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_gan['bounds']['it_pos_now']
        ix1 = data_gan['bounds']['it_max']

        ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, time.time() - start_time))

        if data_gan['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    lang_stats, scores_each, cache_path = language_eval('coco', predictions, eval_kwargs['id'], 'test')

    if netG_rel is not None: netG_rel.train()
    if netG_obj is not None: netG_obj.train()
    if netG_atr is not None: netG_atr.train()

    return lang_stats


def eval_split_g2t_pseudo(opt, model, netG_obj, netG_rel, netG_atr, loader, loader_i2t, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = 5000
    criterionCycle = torch.nn.L1Loss()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        start_time = time.time()
        data_gan = loader.get_batch(split)

        n = n + loader.batch_size

        tmp_gan = [data_gan['isg_feats'][:, 0, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 1, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['isg_feats'][:, 2, :][np.arange(loader.batch_size) * loader.seq_per_img],
                   data_gan['labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp_gan = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp_gan]
        real_A_obj, real_A_rel, real_A_atr, labels = tmp_gan

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            model.use_isg = 0

            seq_isg = model(real_A_obj, real_A_atr, real_A_atr, opt=eval_kwargs, mode='sample_gan')[0].data

        sents_isg = utils.decode_sequence(loader.get_vocab(), seq_isg, use_ssg=True)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:, 1:], use_ssg=True)
        for k, sent in enumerate(sents_isg):
            entry = {'image_id': data_gan['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data_gan['infos'][k]['file_path']
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_gan['bounds']['it_pos_now']
        ix1 = data_gan['bounds']['it_max']

        ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, time.time() - start_time))

        if data_gan['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    lang_stats, scores_each, cache_path = language_eval('coco', predictions, eval_kwargs['id'], 'test')

    return lang_stats

def eval_split_i2t(opt, model, netG_A_obj, netG_A_rel, netG_A_atr, netG_B_obj, netG_B_rel, netG_B_atr, loader, loader_i2t, split='val'):
    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']
    num_images = 5000
    criterionCycle = torch.nn.L1Loss()

    netG_A_obj.eval()
    netG_A_rel.eval()
    netG_A_atr.eval()

    netG_B_obj.eval()
    netG_B_rel.eval()
    netG_B_atr.eval()

    loader.reset_iterator(split)
    loader_i2t.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        start_time = time.time()
        data_i2t = loader_i2t.get_batch(split)

        n = n + loader.batch_size

        tmp_i2t = [data_i2t['fc_feats'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['isg_rela_matrix'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['isg_rela_masks'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['isg_obj'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['isg_obj_masks'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['isg_attr'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['isg_attr_masks'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['labels'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img],
                   data_i2t['masks'][np.arange(loader_i2t.batch_size) * loader_i2t.seq_per_img]]
        tmp_i2t = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp_i2t]
        fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp_i2t
        ssg_data = {}
        ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
        ssg_data['ssg_rela_masks'] = ssg_rela_masks
        ssg_data['ssg_obj'] = ssg_obj
        ssg_data['ssg_obj_masks'] = ssg_obj_masks
        ssg_data['ssg_attr'] = ssg_attr
        ssg_data['ssg_attr_masks'] = ssg_attr_masks

        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        with torch.no_grad():
            seq_isg = model(None, None, None, fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample_i2t')[0].data
            model.use_isg = 0

        sents_isg = utils.decode_sequence(loader_i2t.get_vocab(), seq_isg, use_ssg=True)
        gt_captions = utils.decode_sequence(loader_i2t.get_vocab(), labels[:, 1:], use_ssg=True)
        for k, sent in enumerate(sents_isg):
            entry = {'image_id': data_i2t['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data_i2t['infos'][k]['file_path']
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data_i2t['bounds']['it_pos_now']
        ix1 = data_i2t['bounds']['it_max']

        ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, time.time() - start_time))

        if data_i2t['bounds']['wrapped']:
            break
        if n >= num_images:
            break

    lang_stats, scores_each, cache_path = language_eval('coco', predictions, eval_kwargs['id'], 'test')

    netG_A_obj.eval()
    netG_A_rel.eval()
    netG_A_atr.eval()

    netG_B_obj.eval()
    netG_B_rel.eval()
    netG_B_atr.eval()

    return lang_stats