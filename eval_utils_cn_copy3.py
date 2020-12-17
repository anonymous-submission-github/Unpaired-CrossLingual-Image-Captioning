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
import io
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import jieba

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='key.json'
# Instantiates a client
from google.cloud import translate_v2 as translate
translate_client = translate.Client()

def language_eval(dataset, preds, model_id, split):
    import sys
    # using caption evaluation scripts from [AIChallenge](https://github.com/AIChallenger/AI_Challenger_2017/tree/master/Evaluation/caption_eval/coco_caption)
    # because it's indexed by filename, not image id
    sys.path.append("coco-cn-cap-eval")
    from evals.coco_caption.pycxtools.coco import COCO
    from evals.coco_caption.pycxevalcap.eval import COCOEvalCap

    annFile = 'data/coco_cn/COCOCN_val_test_isg_v2.json'   # test ann contain 6 sentences


    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '_isg_v3_small.json')
    cache_path_1 = os.path.join('eval_results/', model_id + '_' + split +'_scores'+'_isg_v3_small.json')
    # cache_path = os.path.join('eval_results/', model_id + '_' + split + '_ssg_v3_small.json')
    # cache_path_1 = os.path.join('eval_results/', model_id + '_' + split +'_scores'+'_ssg_v3_small.json')

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

    return out, imgToEval

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    # num_images=150
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    output_cap_path = eval_kwargs.get('output_cap_path', '0')

    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path,allow_pickle=True)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']


    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            fc_feats = None
            att_feats = None
            att_masks = None
            ssg_data = None
            rela_data = None

            # forward the model to get loss
            if use_rela == 1:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],
                       data['rela_matrix'], data['rela_masks'], data['attr_matrix'], data['attr_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, rela_matrix, rela_masks, attr_matrix, attr_masks = tmp
                rela_data = {}
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data['rela_matrix'] = rela_matrix
                rela_data['rela_masks'] = rela_masks
                rela_data['attr_matrix'] = attr_matrix
                rela_data['attr_masks'] = attr_masks
            elif use_ssg == 1:
                tmp = [data['fc_feats'], data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'],
                       data['ssg_obj_masks'],
                       data['ssg_attr'], data['ssg_attr_masks'], data['labels'], data['masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                ssg_data['ssg_rela_masks'] = ssg_rela_masks
                ssg_data['ssg_obj'] = ssg_obj
                ssg_data['ssg_obj_masks'] = ssg_obj_masks
                ssg_data['ssg_attr'] = ssg_attr
                ssg_data['ssg_attr_masks'] = ssg_attr_masks
            else:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                # loss = crit(model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data), labels[:,1:], masks[:,1:]).item()
                loss = crit(model(fc_feats, labels, rela_data, ssg_data), labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats = None
        att_feats = None
        att_masks = None
        isg_data = None
        ssg_data = None
        rela_data = None

        if use_ssg == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
            ssg_data = {}
            ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
            ssg_data['ssg_rela_masks'] = ssg_rela_masks
            ssg_data['ssg_obj'] = ssg_obj
            ssg_data['ssg_obj_masks'] = ssg_obj_masks
            ssg_data['ssg_attr'] = ssg_attr
            ssg_data['ssg_attr_masks'] = ssg_attr_masks
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, labels, masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data

        #Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0),
                                                       use_ssg=use_ssg)[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=use_ssg)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=use_ssg)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            # if eval_kwargs.get('dump_path', 0) == 1:
            if eval_kwargs.get('dump_path', 1) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap_'+ eval_kwargs['id'] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats

def eval_split_fc(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    # num_images=150
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    output_cap_path = eval_kwargs.get('output_cap_path', '0')

    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path,allow_pickle=True)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']


    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            fc_feats = None
            att_feats = None
            att_masks = None
            ssg_data = None
            rela_data = None

            # forward the model to get loss
            if use_rela == 1:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],
                       data['rela_matrix'], data['rela_masks'], data['attr_matrix'], data['attr_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, rela_matrix, rela_masks, attr_matrix, attr_masks = tmp
                rela_data = {}
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data['rela_matrix'] = rela_matrix
                rela_data['rela_masks'] = rela_masks
                rela_data['attr_matrix'] = attr_matrix
                rela_data['attr_masks'] = attr_masks
            elif use_ssg == 1:
                tmp = [data['fc_feats'], data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'],
                       data['ssg_obj_masks'],
                       data['ssg_attr'], data['ssg_attr_masks'], data['labels'], data['masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                ssg_data['ssg_rela_masks'] = ssg_rela_masks
                ssg_data['ssg_obj'] = ssg_obj
                ssg_data['ssg_obj_masks'] = ssg_obj_masks
                ssg_data['ssg_attr'] = ssg_attr
                ssg_data['ssg_attr_masks'] = ssg_attr_masks
            else:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp
            with torch.no_grad():
                # loss = crit(model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data), labels[:,1:], masks[:,1:]).item()
                # loss = crit(model(fc_feats, labels, rela_data, ssg_data), labels[:, 1:], masks[:, 1:]).item()
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats = None
        att_feats = None
        att_masks = None
        isg_data = None
        ssg_data = None
        rela_data = None

        if use_ssg == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
            ssg_data = {}
            ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
            ssg_data['ssg_rela_masks'] = ssg_rela_masks
            ssg_data['ssg_obj'] = ssg_obj
            ssg_data['ssg_obj_masks'] = ssg_obj_masks
            ssg_data['ssg_attr'] = ssg_attr
            ssg_data['ssg_attr_masks'] = ssg_attr_masks
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, labels, masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            # seq = model(fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data
            seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

        #Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0),
                                                       use_ssg=use_ssg)[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=use_ssg)
        # gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=use_ssg)

        for k, sent in enumerate(sents):
            # sent = translate_client.translate(sent, target_language='zh-cn')['translatedText']
            # sent = jieba.lcut(sent.rstrip('\r\n'),cut_all=False)
            # sent = ' '.join(sent)
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            # entry['gt_caption'] = gt_captions[k]
            # if eval_kwargs.get('dump_path', 0) == 1:
            if eval_kwargs.get('dump_path', 1) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap_'+ eval_kwargs['id'] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


def eval_split_nmt(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    # num_images = 150
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    output_cap_path = eval_kwargs.get('output_cap_path', '0')

    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path,allow_pickle=True)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']


    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        ssg_data = None
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp

        ssg_data = {}
        ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
        ssg_data['ssg_rela_masks'] = ssg_rela_masks
        ssg_data['ssg_obj'] = ssg_obj
        ssg_data['ssg_obj_masks'] = ssg_obj_masks
        ssg_data['ssg_attr'] = ssg_attr
        ssg_data['ssg_attr_masks'] = ssg_attr_masks

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data

        #Print beam search
        for i in range(loader.batch_size):
            print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0),
                                                   use_ssg=use_ssg)[0] for _ in model.done_beams[i]]))
            print('--' * 10)

        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=use_ssg)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=use_ssg)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap_'+ eval_kwargs['id'] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats

def eval_split_up(model, use_mapping, netG_obj, netG_rel, netG_atr, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    gan_type = eval_kwargs.get('gan_type', 0)
    freeze_i2t = eval_kwargs.get('freeze_i2t', 0)
    output_cap_path = eval_kwargs.get('output_cap_path', '0')

    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']


    # Make sure in the evaluation mode
    model.eval()

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
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            fc_feats = None
            att_feats = None
            att_masks = None
            ssg_data = None
            rela_data = None

            # forward the model to get loss
            if use_rela == 1:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],
                       data['rela_matrix'], data['rela_masks'], data['attr_matrix'], data['attr_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, rela_matrix, rela_masks, attr_matrix, attr_masks = tmp
                rela_data = {}
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data['rela_matrix'] = rela_matrix
                rela_data['rela_masks'] = rela_masks
                rela_data['attr_matrix'] = attr_matrix
                rela_data['attr_masks'] = attr_masks
            elif use_ssg == 1:
                tmp = [data['fc_feats'], data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'],
                       data['ssg_obj_masks'],
                       data['ssg_attr'], data['ssg_attr_masks'], data['labels'], data['masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                ssg_data['ssg_rela_masks'] = ssg_rela_masks
                ssg_data['ssg_obj'] = ssg_obj
                ssg_data['ssg_obj_masks'] = ssg_obj_masks
                ssg_data['ssg_attr'] = ssg_attr
                ssg_data['ssg_attr_masks'] = ssg_attr_masks
            else:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp

            #with torch.no_grad():
                # loss = crit(model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data), labels[:,1:], masks[:,1:]).item()
            #    loss = crit(model(fc_feats, labels, rela_data, ssg_data), labels[:, 1:], masks[:, 1:]).item()
            #loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats = None
        att_feats = None
        att_masks = None
        isg_data = None
        ssg_data = None
        rela_data = None

        if use_ssg == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
            ssg_data = {}
            ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
            ssg_data['ssg_rela_masks'] = ssg_rela_masks
            ssg_data['ssg_obj'] = ssg_obj
            ssg_data['ssg_obj_masks'] = ssg_obj_masks
            ssg_data['ssg_attr'] = ssg_attr
            ssg_data['ssg_attr_masks'] = ssg_attr_masks
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, labels, masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if use_mapping:
                if gan_type == 0:
                    seq = model(netG_obj, netG_rel, netG_atr, fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data
                elif gan_type == 1:
                    seq = model(netG_obj, netG_rel, None, fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data
                elif gan_type == 2:
                    seq = model(netG_obj, None, None, fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data
            else:
                seq = model(None, None, None, fc_feats, ssg_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data

        #Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0),
                                                       use_ssg=use_ssg)[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=use_ssg)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=use_ssg)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap_'+ eval_kwargs['id'] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()

    # Switch back to training mode
    if freeze_i2t == 0:
        model.train()

    if netG_rel is not None: netG_rel.train()
    if netG_obj is not None: netG_obj.train()
    if netG_atr is not None: netG_atr.train()

    return loss_sum/loss_evals, predictions, lang_stats



def eval_split_isg(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    gan_type = eval_kwargs.get('gan_type', 0)
    freeze_i2t = eval_kwargs.get('freeze_i2t', 0)
    output_cap_path = eval_kwargs.get('output_cap_path', '0')

    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']


    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['isg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['isg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['isg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['isg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['isg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['isg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, isg_rela_matrix, isg_rela_masks, isg_obj, isg_obj_masks, isg_attr, isg_attr_masks, labels, masks = tmp
        isg_data = {}

        isg_data['isg_rela_matrix'] = isg_rela_matrix
        isg_data['isg_rela_masks'] = isg_rela_masks
        isg_data['isg_obj'] = isg_obj
        isg_data['isg_obj_masks'] = isg_obj_masks
        isg_data['isg_attr'] = isg_attr
        isg_data['isg_attr_masks'] = isg_attr_masks

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, isg_data, opt=eval_kwargs, mode='sample')[0].data

        #Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0),
                                                       use_ssg=use_ssg)[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=use_ssg)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=use_ssg)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap_'+ eval_kwargs['id'] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()

    # Switch back to training mode
    model.train()

    return loss_sum/loss_evals, predictions, lang_stats


def eval_split_raw(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    output_cap_path = eval_kwargs.get('output_cap_path', '0')

    rela_dict_path = 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']


    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            fc_feats = None
            att_feats = None
            att_masks = None
            ssg_data = None
            rela_data = None

            # forward the model to get loss
            if use_rela == 1:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],
                       data['rela_matrix'], data['rela_masks'], data['attr_matrix'], data['attr_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks, rela_matrix, rela_masks, attr_matrix, attr_masks = tmp
                rela_data = {}
                rela_data['att_feats'] = att_feats
                rela_data['att_masks'] = att_masks
                rela_data['rela_matrix'] = rela_matrix
                rela_data['rela_masks'] = rela_masks
                rela_data['attr_matrix'] = attr_matrix
                rela_data['attr_masks'] = attr_masks
            elif use_ssg == 1:
                tmp = [data['fc_feats'], data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'],
                       data['ssg_obj_masks'],
                       data['ssg_attr'], data['ssg_attr_masks'], data['labels'], data['masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                ssg_data['ssg_rela_masks'] = ssg_rela_masks
                ssg_data['ssg_obj'] = ssg_obj
                ssg_data['ssg_obj_masks'] = ssg_obj_masks
                ssg_data['ssg_attr'] = ssg_attr
                ssg_data['ssg_attr_masks'] = ssg_attr_masks
            else:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats = None
        att_feats = None
        att_masks = None
        ssg_data = None
        rela_data = None

        if use_rela == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['attr_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]
                   ]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, rela_matrix, rela_masks, attr_matrix, attr_masks,\
                labels, masks = tmp
            rela_data = {}
            rela_data['att_feats'] = att_feats
            rela_data['att_masks'] = att_masks
            rela_data['rela_matrix'] = rela_matrix
            rela_data['rela_masks'] = rela_masks
            rela_data['attr_matrix'] = attr_matrix
            rela_data['attr_masks'] = attr_masks
        elif use_ssg == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
            ssg_data = {}
            ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
            ssg_data['ssg_rela_masks'] = ssg_rela_masks
            ssg_data['ssg_obj'] = ssg_obj
            ssg_data['ssg_obj_masks'] = ssg_obj_masks
            ssg_data['ssg_attr'] = ssg_attr
            ssg_data['ssg_attr_masks'] = ssg_attr_masks
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, labels, masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, rela_data, ssg_data, opt=eval_kwargs, mode='sample')[0].data

        #Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0),
                                                       use_ssg=use_ssg)[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq, use_ssg=use_ssg)
        gt_captions = utils.decode_sequence(loader.get_vocab(), labels[:,1:], use_ssg=use_ssg)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['gt_caption'] = gt_captions[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)


            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(dataset, predictions, eval_kwargs['id'], split)

        if verbose:
            for img_id in scores_each.keys():
                print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file = open('gen_cap/cap'+ eval_kwargs['model'][-10:-4] + '.txt', "aw")
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
                text_file.close()

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


def extract_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_rela = eval_kwargs.get('use_rela', 0)
    use_ssg = eval_kwargs.get('use_ssg', 0)
    which_to_extract = eval_kwargs.get('which_to_extract', 'e')


    root_path = '/home/yangxu/project/self-critical.pytorch/'
    rela_dict_path = root_path + 'data/rela_dict.npy'
    rela_dict_info = np.load(rela_dict_path)
    rela_dict_info = rela_dict_info[()]
    rela_dict = rela_dict_info['rela_dict']

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    save_t = 0
    save_id = 1
    rnn_size = 1000
    N_save = 50
    print('total num to save {0}'.format(int(110000)/(N_save*loader.batch_size)))
    if which_to_extract == 'h':
        save_path = 'save_hs'+ '/' + eval_kwargs.get('checkpoint_path', 0) + '/'
    elif which_to_extract == 'e':
        save_path = 'save_e' + '/' + eval_kwargs.get('checkpoint_path', 0) + '/'
    data_save = np.zeros([0, rnn_size])

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        save_t = save_t + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats = None
        att_feats = None
        att_masks = None
        ssg_data = None
        rela_data = None

        if use_rela == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['attr_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]
                   ]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, rela_matrix, rela_masks, attr_matrix, attr_masks, \
            labels, masks = tmp
            rela_data = {}
            rela_data['att_feats'] = att_feats
            rela_data['att_masks'] = att_masks
            rela_data['rela_matrix'] = rela_matrix
            rela_data['rela_masks'] = rela_masks
            rela_data['attr_matrix'] = attr_matrix
            rela_data['attr_masks'] = attr_masks
        elif use_ssg == 1:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_matrix'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_rela_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_obj_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['ssg_attr_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
            ssg_data = {}
            ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
            ssg_data['ssg_rela_masks'] = ssg_rela_masks
            ssg_data['ssg_obj'] = ssg_obj
            ssg_data['ssg_obj_masks'] = ssg_obj_masks
            ssg_data['ssg_attr'] = ssg_attr
            ssg_data['ssg_attr_masks'] = ssg_attr_masks
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks, labels, masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if which_to_extract == 'h':
                data_temp = model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data, mode='extract_hs')
                data_temp = data_temp.cpu().numpy()
                mask_temp = masks.cpu().numpy()
            elif which_to_extract == 'e':
                data_temp, mask_temp = model(fc_feats, att_feats, labels, att_masks, rela_data, ssg_data, mode='extract_e')
                data_temp = data_temp.cpu().numpy()
                mask_temp = mask_temp.cpu().numpy()

            data_size = np.shape(data_temp)
            data_temp = np.reshape(data_temp, (-1,data_size[-1]))
            mask_temp = np.reshape(mask_temp, (np.shape(data_temp)[0],))
            index = np.where(mask_temp == 1)
            data_use = data_temp[index]
            data_save = np.concatenate((data_save,data_use),axis = 0)
            print('save_t is {0}'.format(save_t))

            if save_t % N_save == 0:
                save_path_temp = save_path + format(int(save_t/N_save), '04') + '.npz'
                np.savez(save_path_temp, saved_data=data_save)
                data_save = np.zeros([0, rnn_size])

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break






