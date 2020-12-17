from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, use_ssg = False):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                if use_ssg:
                    txt = txt + ix_to_word[ix.item()]
                else:
                    txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class SgCriterion_weak(nn.Module):
    def __init__(self):
        super(SgCriterion_weak, self).__init__()

    def forward(self, input, target):
        """
        :param input: batch_size* class_size
        :param target: batch_size* class_size
        :return: loss
        """
        output = torch.log(target*(input-0.5)+0.5)
        output = -torch.sum(output)/input.size(0)
        return output

class SgCriterion(nn.Module):
    def __init__(self):
        super(SgCriterion, self).__init__()

    def forward(self, input, target, mask, train_mode):
        """
        :param input: batch_size* max_att* class_size
        :param target: batch_size* max_att
               mask: batch_size*max_att
        :return: loss
        """
        if train_mode == 'attr':
            input_inv = 1-input
            target_inv = 1-target
            output = -(torch.log(input)*target.float() + torch.log(input_inv)*target_inv.float())
            output = torch.sum(output,dim=2)
            output = torch.sum(output*mask)/torch.sum(mask)
            #target: batch_size * max_att *class_size
            #input: batch_size* max_att* class_size
        else:
            input_temp = -input.gather(2, target.long().unsqueeze(2)).squeeze(2)
            output = torch.sum(input_temp*mask)/torch.sum(mask)
        return output

def SgMae(input, target, mask, top, train_mode):
    """
    :param input: batch_size* max_att* class_size
    :param target: batch_size* max_att
           mask: batch_size*max_att
    :return: loss
    """
    sort_value, sort_index = torch.sort(input, dim=2, descending=True)
    sort_index = sort_index.cpu().numpy()
    top_objs = sort_index[:, :, :top]
    data_size = input.size()
    output = 0
    num_label = 0
    if train_mode == 'attr':
        for batch_id in range(data_size[0]):
            box_len = np.sum(mask[batch_id])
            for j in range(box_len):
                attr_label = np.where(target[batch_id,j] == 1)[0]
                attr_len = len(attr_label)
                for i in range(attr_len):
                    index = 0
                    num_label += attr_len
                    for k in range(top):
                        if attr_label[i] == top_objs[batch_id,j,k]:
                            index = 1
                    output += index
    else:
        for batch_id in range(data_size[0]):
            box_len = np.sum(mask[batch_id])
            num_label += box_len
            for j in range(box_len):
                for k in range(top):
                    if top_objs[batch_id,j,k] == target[batch_id,j]:
                        output += 1

    return output, num_label, top_objs

def SgPred(input, top, train_mode):
    """
    :param input: batch_size* max_att* class_size
    :param target: batch_size* max_att
           mask: batch_size*max_att
    :return: loss
    """
    sort_value, sort_index = torch.sort(input, dim=2, descending=True)
    sort_index = sort_index.cpu().numpy()
    top_objs = sort_index[:, :, :top]
    return top_objs

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if type(param.grad) != type(None):
                param.grad.data.clamp_(-grad_clip, grad_clip)

def compute_diff(diff, mask):
    output = torch.sum(diff*mask)/torch.sum(mask)
    return output

def compute_dis_diff(dis1, dis2, mask):
    loss_temp = torch.sum((dis1 - dis2).pow(2), 2)/dis1.size(2)
    loss = torch.sum(loss_temp*mask)/torch.sum(mask)
    return loss

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def build_optimizer(params, opt, optim_method='adam'):
    if optim_method == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif optim_method == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif optim_method == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif optim_method == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif optim_method == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif optim_method == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def unison_shuffled_isg_copies(data):
    p = np.random.permutation(len(data['fc_feats']))
    data['isg_rela_matrix'] = data['isg_rela_matrix'][p]
    data['isg_rela_masks'] = data['isg_rela_masks'][p]
    data['isg_obj'] = data['isg_obj'][p]
    data['isg_obj_masks'] = data['isg_obj_masks'][p]
    data['isg_attr'] = data['isg_attr'][p]
    data['isg_attr_masks'] = data['isg_attr_masks'][p]
    return data

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images