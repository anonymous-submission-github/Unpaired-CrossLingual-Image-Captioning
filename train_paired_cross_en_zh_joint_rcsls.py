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
import opts_bi
import models
from dataloader_up_mt_crosslingual import *
# import eval_utils_cn
# import eval_utils_en
import misc.utils as utils
from misc.rewards_up import init_scorer, get_self_critical_reward
from models.weight_init import Model_init
import torch.nn.functional as F




try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, keys, value, iteration):
    summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=keys, simple_value=value)])
    writer.add_summary(summary, iteration)

def model_start(start_from,p_flag):
    # checkpoint_path = start_from
    # id = checkpoint_path.split('/')[-1]
    # print('Point to folder: {}'.format(checkpoint_path))
    # return checkpoint_path,id

    if start_from is not None:
        checkpoint_path = start_from
        id = checkpoint_path.split('/')[-1]
        print('Point to folder: {}'.format(checkpoint_path))
    else:
        time.sleep(5)

        if p_flag==0:
            id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + opt.caption_model_zh
            checkpoint_path = os.path.join('save_for_joint', id)
        else:
            id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + opt.caption_model_en
            checkpoint_path = os.path.join('save_for_joint', id)

        if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
        print('Create folder: {}'.format(checkpoint_path))

    return checkpoint_path,id


def load_info(loader,start_from,checkpoint_path,p_flag):
    infos = {}
    histories = {}
    if start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(checkpoint_path, 'infos.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            # need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            need_be_same = ["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(checkpoint_path, 'histories.pkl')):
            with open(os.path.join(checkpoint_path, 'histories.pkl')) as f:
                histories = cPickle.load(f)

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

    opt.p_flag=p_flag
    if getattr(opt, 'p_flag', 0) == 0:
        opt.caption_model=opt.caption_model_zh
    else:
        opt.caption_model=opt.caption_model_en

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
    train_loss_kl=0
    train_loss_all=0

    reward = np.zeros([1, 1])
    return loader,iteration,epoch,val_result_history,loss_history,lr_history,ss_prob_history,best_val_score,\
           infos,histories,update_lr_flag,model,dp_model,parameters,crit,rl_crit,optimizer,accumulate_iter,train_loss,reward,train_loss_kl,train_loss_all

def pre_model(update_lr_flag,epoch,optimizer,model,data,dp_model,crit,rl_crit,p_flag):


    sc_flag=False

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


    fc_feats = None
    att_feats = None
    att_masks = None
    ssg_data = None
    rela_data = None

    if getattr(opt, 'use_ssg', 0) == 1:
        if getattr(opt, 'use_isg', 0) == 1:
            tmp = [data['fc_feats'], data['labels'], data['masks'], data['att_feats'], data['att_masks'],
                   data['isg_rela_matrix'], data['isg_rela_masks'], data['isg_obj'], data['isg_obj_masks'],
                   data['isg_attr'], data['isg_attr_masks'],
                   data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'], data['ssg_obj_masks'],
                   data['ssg_attr'], data['ssg_attr_masks']]

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
            if p_flag == 1:
                tmp = [data['fc_feats'], data['ssg_paired_rela_matrix'], data['ssg_paired_rela_masks'], data['ssg_paired_obj'],data['ssg_paired_obj_masks'],
                       data['ssg_paired_attr'], data['ssg_paired_attr_masks'], data['labels_p'], data['masks_p']]
                # print (tmp)
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
                tmp = [data['fc_feats'], data['ssg_rela_matrix'], data['ssg_rela_masks'], data['ssg_obj'],data['ssg_obj_masks'],
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
        outputs, att_obj,att_rela,att_attr = dp_model(fc_feats, labels, isg_data, ssg_data)
        loss = crit(outputs, labels[:, 1:], masks[:, 1:])
    else:
        gen_result, sample_logprobs = dp_model(fc_feats, isg_data, ssg_data, opt={'sample_max': 0}, mode='sample')
        reward = get_self_critical_reward(dp_model, fc_feats, isg_data, ssg_data, data, gen_result, opt)
        loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

    return loss,att_obj,att_rela,att_attr,update_lr_flag,sc_flag


def save_model(input_json,accumulate_iter,optimizer,iteration,loss,sc_flag,epoch,start,reward,data,tb_summary_writer,model,
               loss_history,lr_history,ss_prob_history,use_rela,dp_model,val_result_history,best_val_score,
               crit,loader,infos,histories,train_loss,train_loss_kl ,train_loss_all,id,opt_checkpoint_path,p_flag,loss_all,loss_kl,update_lr_flag):

    if p_flag==0:
        import eval_utils_cn as eval_utils
    else:
        import eval_utils_en_copy as eval_utils

    if accumulate_iter % opt.accumulate_number == 0:
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        iteration += 1
        accumulate_iter = 0
        train_loss = loss.item() * opt.accumulate_number
        train_loss_kl=loss_kl.item()* opt.accumulate_number
        train_loss_all=loss_all.item()* opt.accumulate_number
        end = time.time()

        if not sc_flag:
            print("{}/{}/{}|train_loss={:.3f}|weighted train_loss_kl={:.3f}|time/batch={:.3f}" \
                  .format(id, iteration, epoch, train_loss, train_loss_kl, end - start))
        else:
            print("{}/{}/{}|avg_reward={:.3f}|time/batch={:.3f}" \
                  .format(id, iteration, epoch, np.mean(reward[:, 0]), end - start))

    torch.cuda.synchronize()

    # Update the iteration and epoch
    if data['bounds']['wrapped']:
        epoch += 1
        update_lr_flag = True

    # Write the training loss summary
    if (iteration % opt.losses_log_every == 0) and (iteration != 0):
        add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
        add_summary_value(tb_summary_writer, 'weighted train_loss_kl', train_loss_kl, iteration)
        add_summary_value(tb_summary_writer, 'weighted train_loss_all', train_loss_all, iteration)
        add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
        add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
        if sc_flag:
            add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)

        loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
        lr_history[iteration] = opt.current_lr
        ss_prob_history[iteration] = model.ss_prob

    # make evaluation on validation set, and save model
    # if (iteration %10 == 0) and (iteration != 0):
    if (iteration % opt.save_checkpoint_every == 0) and (iteration != 0):
        # eval model
        if use_rela:
            eval_kwargs = {'split': 'val',
                           'dataset': input_json,
                           'use_real': 1}
        else:
            eval_kwargs = {'split': 'val',
                           'dataset': input_json}
        eval_kwargs.update(vars(opt))
        val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, p_flag, eval_kwargs)

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
            checkpoint_path = os.path.join(opt_checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt_checkpoint_path, 'optimizer.pth')
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
            with open(os.path.join(opt_checkpoint_path, 'infos.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt_checkpoint_path, 'histories.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            if best_flag:
                checkpoint_path = os.path.join(opt_checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt_checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

    return update_lr_flag,epoch,optimizer,model,dp_model,accumulate_iter,\
                   iteration, loss, sc_flag, start, reward, tb_summary_writer,\
                   loss_history, lr_history, ss_prob_history, use_rela, val_result_history, best_val_score,\
                   loader, infos, histories,train_loss, train_loss_kl ,train_loss_all,loss_all,loss_kl

def train(opt):
    start_from=vars(opt).get('start_from', None)
    start_from_p=vars(opt).get('start_from_en', None)
    opt.checkpoint_path, opt.id= model_start(start_from,0)
    opt.checkpoint_path_p, opt.id_p= model_start(start_from_p,1)

    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    # opt.use_att = False
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = DataLoader_UP(opt)
    vocab_size = loader.vocab_size
    vocab_size_p= loader.vocab_size_p

    if opt.use_rela == 1:
        opt.rela_dict_size = loader.rela_dict_size
    opt.seq_length = loader.seq_length
    use_rela = getattr(opt, 'use_rela', 0)

    try:
        tb_summary_writer = tf and tf.compat.v1.summary.FileWriter(opt.checkpoint_path)
        tb_summary_writer_p = tf and tf.compat.v1.summary.FileWriter(opt.checkpoint_path_p)
    except:
        print('Set tensorboard error!')
        pdb.set_trace()

    opt.p_flag=0 # whether paired model
    opt.vocab_size=vocab_size
    loader,iteration,epoch,val_result_history,loss_history,lr_history,ss_prob_history,best_val_score,\
    infos,histories,update_lr_flag,model,dp_model,parameters,crit,rl_crit,optimizer,accumulate_iter,train_loss,reward,train_loss_kl,train_loss_all=load_info(loader,start_from,opt.checkpoint_path,opt.p_flag)

    opt.p_flag =1
    opt.vocab_size=vocab_size_p
    loader,iteration_p,epoch_p,val_result_history_p,loss_history_p,lr_history_p,ss_prob_history_p,best_val_score_p, \
    infos_p, histories_p, update_lr_flag_p, model_p,dp_model_p, parameters_p, crit_p, rl_crit_p, optimizer_p, accumulate_iter_p, train_loss_p, reward_p,train_loss_kl,train_loss_all = load_info(
        loader, start_from_p,opt.checkpoint_path_p,opt.p_flag)

# #  global variables
    update_lr_flag=update_lr_flag
    accumulate_iter = accumulate_iter
    train_loss = train_loss
    train_loss_kl= train_loss_kl
    train_loss_all= train_loss_all
    reward = reward

    update_lr_flag_p=update_lr_flag_p
    accumulate_iter_p = accumulate_iter_p
    train_loss_p = train_loss_p
    reward_p = reward_p


    while True:

        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)
        # print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        opt.p_flag = 0  # whether paired model
        loss,att_obj,att_rela,att_attr,update_lr_flag,sc_flag=pre_model(update_lr_flag,epoch,optimizer,model,data,dp_model,crit,rl_crit,opt.p_flag)

        opt.p_flag= 1
        loss_p,att_obj_p,att_rela_p,att_attr_p,update_lr_flag_p,sc_flag_p=pre_model(update_lr_flag_p,epoch_p,optimizer_p,model_p,data,dp_model_p,crit_p,rl_crit_p,opt.p_flag)

        att_obj = F.softmax(att_obj, dim=1)
        att_obj_p = F.softmax(att_obj_p, dim=1)
        att_rela = F.softmax(att_rela, dim=1)
        att_rela_p = F.softmax(att_rela_p, dim=1)
        att_attr = F.softmax(att_attr, dim=1)
        att_attr_p = F.softmax(att_attr_p, dim=1)

        loss_kl = torch.exp(F.kl_div(att_obj.log(), att_obj_p, reduction='sum'),out=None)\
                  +torch.exp(F.kl_div(att_rela.log(), att_rela_p, reduction='sum'),out=None)\
                  +torch.exp(F.kl_div(att_attr.log(), att_attr_p, reduction='sum'),out=None)
        # print(loss_kl)

        accumulate_iter = accumulate_iter + 1
        accumulate_iter_p = accumulate_iter_p + 1
        loss_all=loss+loss_p+loss_kl
        loss = loss / opt.accumulate_number
        loss_p = loss_p / opt.accumulate_number
        loss_kl=loss_kl/opt.accumulate_number
        loss_all = loss_all / opt.accumulate_number
        loss_all.backward()


        opt.p_flag = 0
        # print ('iteration of model 1 is {}'.format(iteration))
        update_lr_flag, epoch, optimizer, model, dp_model, accumulate_iter, iteration, loss, sc_flag, start, reward, \
        tb_summary_writer, loss_history, lr_history, ss_prob_history, use_rela, val_result_history, best_val_score, loader,\
        infos, histories, train_loss, train_loss_kl, train_loss_all, loss_all, loss_kl=\
        save_model(opt.input_json,accumulate_iter, optimizer, iteration, loss, sc_flag, epoch, start, reward, data, tb_summary_writer,model,
                   loss_history, lr_history, ss_prob_history, use_rela, dp_model, val_result_history, best_val_score,
                   crit, loader, infos, histories,train_loss, train_loss_kl ,train_loss_all,opt.id,opt.checkpoint_path,opt.p_flag,loss_all,loss_kl,update_lr_flag)

        opt.p_flag = 1
        # print('iteration of model 2 is {}'.format(iteration_p))
        update_lr_flag_p, epoch_p, optimizer_p, model_p, dp_model_p, accumulate_iter_p, iteration_p, loss_p, sc_flag_p, start, reward_p, \
        tb_summary_writer_p, loss_history_p, lr_history_p, ss_prob_history_p, use_rela, val_result_history_p, best_val_score_p, loader,\
        infos_p, histories_p, train_loss_p, train_loss_kl, train_loss_all, loss_all, loss_kl=\
        save_model(opt.input_json_en,accumulate_iter_p, optimizer_p, iteration_p, loss_p, sc_flag_p, epoch_p, start, reward_p, data, tb_summary_writer_p, model_p,
                   loss_history_p, lr_history_p, ss_prob_history_p, use_rela, dp_model_p, val_result_history_p, best_val_score_p,
                   crit_p, loader, infos_p, histories_p,train_loss_p,train_loss_kl ,train_loss_all,opt.id_p,opt.checkpoint_path_p,opt.p_flag,loss_all,loss_kl,update_lr_flag_p)


        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break



opt = opts.parse_opt()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
train(opt)

