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

import numpy as np
import pdb
import time
import os
from six.moves import cPickle

# import opts
import opts_en as opts
import models
from dataloader_up_mt import *
import eval_utils_en_fc as eval_utils
import misc.utils as utils
from misc.rewards_up import init_scorer, get_self_critical_reward
from models.weight_init import Model_init
import  argparse
import json




try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, keys, value, iteration):
    summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=keys, simple_value=value)])
    writer.add_summary(summary, iteration)


def train(opt):
    if vars(opt).get('start_from_en', None) is not None:
        opt.checkpoint_path_p = opt.start_from_en
        opt.id_p = opt.checkpoint_path_p.split('/')[-1]
        print('Point to folder: {}'.format(opt.checkpoint_path_p))
    else:
        opt.id_p = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + opt.caption_model
        opt.checkpoint_path_p = os.path.join(opt.checkpoint_path_p, opt.id_p)

        if not os.path.exists(opt.checkpoint_path_p): os.makedirs(opt.checkpoint_path_p)
        print('Create folder: {}'.format(opt.checkpoint_path_p))

    # # Deal with feature things before anything
    # opt.use_att = utils.if_use_att(opt.caption_model)
    # if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    # loader = DataLoader_UP(opt)
    # opt.vocab_size = loader.vocab_size
    # if opt.use_rela == 1:
    #     opt.rela_dict_size = loader.rela_dict_size
    # opt.seq_length = loader.seq_length
    # use_rela = getattr(opt, 'use_rela', 0)

    try:
        tb_summary_writer = tf and tf.compat.v1.summary.FileWriter(opt.checkpoint_path_p)
    except:
        print('Set tensorboard error!')
        pdb.set_trace()

    infos = {}
    histories = {}
    if opt.start_from_en is not None or opt.use_pretrained_setting==1:
        # open old infos and check if models are compatible
        # with open(os.path.join(opt.checkpoint_path_p, 'infos.pkl')) as f:
        #     infos = cPickle.load(f)
        #     saved_model_opt = infos['opt']
        #     need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
        #     for checkme in need_be_same:
        #         assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        #
        #     # override and collect parameters
        #     if len(opt.input_fc_dir) == 0:
        #         opt.input_fc_dir = infos['opt'].input_fc_dir
        #         opt.input_att_dir = infos['opt'].input_att_dir
        #         opt.input_box_dir = infos['opt'].input_box_dir
        #         # opt.input_label_h5 = infos['opt'].input_label_h5
        #     if len(opt.input_json) == 0:
        #         opt.input_json = infos['opt'].input_json
        #     if opt.batch_size == 0:
        #         opt.batch_size = infos['opt'].batch_size
        #     if len(opt.id) == 0:
        #         opt.id = infos['opt'].id
        #         # opt.id = infos['opt'].id_p
        #
        #     ignore = ['checkpoint_path', "use_gfc", "use_isg", "ssg_dict_path", "input_json", "input_label_h5", "id",
        #               "batch_size", "start_from", "language_eval", "use_rela", "input_ssg_dir", "ssg_dict_path",
        #               "input_rela_dir", "use_spectral_norm", "beam_size", 'gpu', 'caption_model','use_att','max_epochs']
        #     beam_size = opt.beam_size
        #
        #     vocab = infos['vocab']  # ix -> word mapping
        #     opt.vocab = vocab
        #     opt.vocab_size = len(vocab)
        #     for k in vars(infos['opt']).keys():
        #         if k != 'model':
        #             if k not in ignore:
        #                 if k in vars(opt):
        #                     # assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        #                     vars(opt).update({k: vars(infos['opt'])[k]})
        #                     print (vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent, will be copyed from pretrained model')
        #                 else:
        #                     vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model
        #     opt.input_fc_dir = 'data/cocobu_fc'
        #     opt.p_flag = 0

        # Load infos
        # opt.infos_path=os.path.join(opt.checkpoint_path_p, 'infos.pkl')
        opt.infos_path=os.path.join('data/fc/infos.pkl')
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
            # opt.id = infos['opt'].id_p

        ignore = ['checkpoint_path', "use_gfc", "use_isg", "ssg_dict_path", "input_json", "input_label_h5", "id",
                  "batch_size", "start_from", "language_eval", "use_rela", "input_ssg_dir", "ssg_dict_path",
                  "input_rela_dir", "use_spectral_norm", "beam_size", 'gpu', 'caption_model','self_critical_after','save_checkpoint_every']
        beam_size = opt.beam_size
        for k in vars(infos['opt']).keys():
            if k != 'model':
                if k not in ignore:
                    if k in vars(opt):
                        if not vars(opt)[k] == vars(infos['opt'])[k]:
                            print (k + ' option not consistent, copyed from pretrained model')
                            vars(opt).update({k: vars(infos['opt'])[k]})
                        else:
                            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

        vocab = infos['vocab']  # ix -> word mapping
        opt.vocab = vocab
        opt.vocab_size = len(vocab)
        opt.input_fc_dir = 'data/cocobu_fc'

        if os.path.isfile(os.path.join(opt.checkpoint_path_p, 'histories.pkl')):
            with open(os.path.join(opt.checkpoint_path_p, 'histories.pkl')) as f:
                histories = cPickle.load(f)

    # Create the Data Loader instance
    loader = DataLoader_UP(opt)
    if opt.use_rela == 1:
        opt.rela_dict_size = loader.rela_dict_size
    opt.seq_length = loader.seq_length
    use_rela = getattr(opt, 'use_rela', 0)
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    try: # if use pretrained model
        loader.ix_to_word = infos['vocab']
    except: # if train from scratch
        infos = json.load(open(opt.input_json))
        opt.ix_to_word = infos['ix_to_word']
        opt.vocab_size = len(opt.ix_to_word)

    # iteration = infos.get('iter', 0)
    # epoch = infos.get('epoch', 0)
    iteration = 0
    epoch = 0

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # Setup the model
    try:
        opt.caption_model = opt.caption_model_zh
    except:
        opt.caption_model = opt.caption_model
    model = models.setup(opt).cuda()
    # dp_model = torch.nn.DataParallel(model)
    # dp_model = torch.nn.DataParallel(model, [0,2,3])
    dp_model = model

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()
    parameters = model.named_children()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), opt)

    optimizer.zero_grad()
    accumulate_iter = 0
    train_loss = 0
    reward = np.zeros([1, 1])

    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)
        # print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        fc_feats = None
        att_feats = None
        att_masks = None
        ssg_data = None
        rela_data = None

        if getattr(opt, 'use_ssg', 0) == 1:
            if getattr(opt, 'use_isg', 0) == 1:
                tmp = [data['fc_feats'], data['labels'], data['masks'], data['att_feats'], data['att_masks'],
                       data['isg_rela_matrix'], data['isg_rela_masks'], data['isg_obj'], data['isg_obj_masks'], data['isg_attr'], data['isg_attr_masks'],
                       data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'], data['ssg_obj_masks'], data['ssg_attr'], data['ssg_attr_masks']]

                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
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
            else:
                tmp = [data['fc_feats'], data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'], data['ssg_obj_masks'],
                       data['ssg_attr'], data['ssg_attr_masks'], data['labels'], data['masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, ssg_rela_matrix, ssg_rela_masks, ssg_obj, ssg_obj_masks, ssg_attr, ssg_attr_masks, labels, masks = tmp
                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                ssg_data['ssg_rela_masks'] = ssg_rela_masks
                ssg_data['ssg_obj'] = ssg_obj
                ssg_data['ssg_obj_masks'] = ssg_obj_masks
                ssg_data['ssg_attr'] = ssg_attr

                isg_data = None
                ssg_data['ssg_attr_masks'] = ssg_attr_masks
        else:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

        if not sc_flag:
            # loss = crit(dp_model(fc_feats, labels, isg_data, ssg_data), labels[:, 1:], masks[:, 1:])
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, isg_data, ssg_data, opt={'sample_max': 0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, isg_data, ssg_data, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        accumulate_iter = accumulate_iter + 1
        loss = loss / opt.accumulate_number
        loss.backward()
        if accumulate_iter % opt.accumulate_number == 0:
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            iteration += 1
            accumulate_iter = 0
            train_loss = loss.item() * opt.accumulate_number
            end = time.time()

            if not sc_flag:
                print("{}/{}/{}|train_loss={:.3f}|time/batch={:.3f}" \
                      .format(opt.id_p, iteration, epoch, train_loss, end - start))
            else:
                print("{}/{}/{}|avg_reward={:.3f}|time/batch={:.3f}" \
                      .format(opt.id_p, iteration, epoch, np.mean(reward[:, 0]), end - start))

        torch.cuda.synchronize()

        # Update the iteration and epoch
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0) and (iteration != 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0) and (iteration != 0):
        # if (iteration % 100 == 0) and (iteration != 0):
            # eval model
            if use_rela:
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json,
                               'use_real': 1}
            else:
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split_fc(dp_model, crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k, v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:  # if true
                save_id = iteration / opt.save_checkpoint_every
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path_p = os.path.join(opt.checkpoint_path_p, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path_p)
                print("model saved to {}".format(checkpoint_path_p))
                optimizer_path = os.path.join(opt.checkpoint_path_p, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

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
                with open(os.path.join(opt.checkpoint_path_p, 'infos.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path_p, 'histories.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path_p = os.path.join(opt.checkpoint_path_p, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path_p)
                    print("model saved to {}".format(checkpoint_path_p))
                    with open(os.path.join(opt.checkpoint_path_p, 'infos-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break



opt = opts.parse_opt()
# opt.input_json='data/COCOCN_en_addtraindata.json'
# opt.input_label_h5='data/COCOCN_en_addtraindata_label.h5'
# opt.input_json='data/COCOCN_en.json'
# opt.input_label_h5='data/COCOCN_en_label.h5'
# opt.gpu=0
# opt.caption_model='newfc'
# opt.use_ssg=0
# opt.use_isg=0
# opt.p_flag=1
# opt.start_from_en='unpaired_image_caption_revise/save/20201101_115812_newfc/'
# opt.checkpoint_path_p='unpaired_image_caption_revise/save/20201101_115812_newfc/'
# opt.rnn_size=1000
# opt.self_critical_after=1000000000
# opt.batch_size=50
# opt.save_checkpoint_every=1000
# opt.seq_per_img=5
# opt.train_split='train'
# opt.use_pretrained_setting=0
# opt.num_images=1000

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
train(opt)

