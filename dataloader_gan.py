from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from models.ass_fun import *

import torch
import torch.utils.data as data

import multiprocessing


class DataLoader_GAN(data.Dataset):

    def reset_iterator(self, split):
        # if load files from directory, then reset the prefetch process
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_rela_dict_size(self):
        return self.rela_dict_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_ssg = getattr(opt, 'use_ssg', 0)
        self.use_isg = getattr(opt, 'use_isg', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.info_isg = json.load(open(self.opt.input_json_isg))

        # # fileter 20341 traning data for cycle gan mapping
        # tmp_info = self.info
        # tmp_images=tmp_info['images']
        # images_20341=[img for img in tmp_images if img['split']=='train'][:20341]
        # tmp_info['images']=images_20341
        # self.info=tmp_info
        #
        # # fliter coco image dataset( the raw json including translation dataset
        # tmp_info=self.info_isg
        # images_coco=[img for img in tmp_info['images'] if 'coco' in img['id']]
        # tmp_info['images']=images_coco
        # self.info_isg=tmp_info

        if self.use_ssg:
            print('using sentence scene graph info')
            ssg_dict_info = np.load(self.opt.ssg_dict_path,allow_pickle=True)['spice_dict'][()]
            # ssg_dict_info = np.load(self.opt.ssg_dict_path,allow_pickle=True)[()]
            self.ix_to_word = ssg_dict_info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            self.input_ssg_dir = self.opt.input_ssg_dir

        if self.use_isg:
            self.input_isg_dir = self.opt.input_isg_dir
            self.rela_dict_dir = self.opt.rela_dict_dir
            rela_dict_info = np.load(self.rela_dict_dir,allow_pickle=True)
            rela_dict = rela_dict_info[()]['rela_dict']
            self.rela_dict_size = len(rela_dict)
            print('rela dict size is {0}'.format(self.rela_dict_size))

        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir,
              opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

        # #  independent training
        # self.input_isg_dir = "data/coco_graph_extract_ft_isg_obj_rela"
        # # self.input_isg_dir = "data/coco_graph_extract_ft_isg"
        # self.input_ssg_dir = "data/coco_graph_extract_ft_ssg"

        # joint training
        # self.input_isg_dir = "data/coco_graph_extract_ft_isg_joint_obj_rela"
        # self.input_isg_dir = "data/coco_graph_extract_ft_isg_joint"
        # self.input_ssg_dir = "data/coco_graph_extract_ft_ssg_joint"
        # self.input_isg_dir = "data/coco_graph_extract_ft_isg_joint_rcsls"
        # self.input_ssg_dir = "data/coco_graph_extract_ft_ssg_joint_rcsls"
        self.input_isg_dir = opt.input_isg_dir
        self.input_ssg_dir = opt.input_ssg_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        print("seq_size:{0}".format(seq_size))
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))


        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': [], 'train_sg': [], 'val_sg': [], 'test_sg': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))
        print('assigned %d images to split train_sg' % len(self.split_ix['train_sg']))
        print('assigned %d images to split val_sg' % len(self.split_ix['val_sg']))
        print('assigned %d images to split test_sg' % len(self.split_ix['test_sg']))

        self.max_train_num = len(self.split_ix['train'])
        self.max_index = len(self.split_ix['train'] + self.split_ix['test'] + self.split_ix['val'])
        self.isg_unorder = np.random.permutation(self.max_index)
        self.ssg_unorder = np.random.permutation(self.max_index)

        self.iterators = {'train': 0, 'val': 0, 'test': 0, 'train_sg': 0, 'val_sg': 0, 'test_sg': 0}

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        if split == "train":
            isg_batch = np.ndarray((batch_size, 3, 1000), dtype = 'float64')
            ssg_batch = np.ndarray((batch_size, 3, 1000), dtype = 'float64')
        else:
            isg_batch = np.ndarray((batch_size * seq_per_img, 3, 1000), dtype='float64')
            ssg_batch = np.ndarray((batch_size * seq_per_img, 3, 1000), dtype='float64')

        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float64')

        max_index = len(split_ix)
        wrapped = False
        infos = []
        gts = []
        for i in range(batch_size):
            import time

            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]

            if split == 'train':
                ix_isg = self.isg_unorder[ri]
                ix_ssg = self.ssg_unorder[ri]
            else:
                ix_isg = ix
                ix_ssg = ix

            # fetch image
            if split == "train":
                # print (os.path.join(self.input_ssg_dir, str(self.info['images'][ix_ssg]['id']) + '.npy'))
                ssg_batch[i] = np.load(os.path.join(self.input_ssg_dir, str(self.info['images'][ix_ssg]['id']) + '.npy'),allow_pickle=True)
                # print (os.path.join(self.input_isg_dir, str(self.info_isg['images'][ix_ssg]['id']) + '.npy'))
                isg_batch[i] = np.load(os.path.join(self.input_isg_dir, str(self.info_isg['images'][ix_isg]['id']) + '.npy'),allow_pickle=True)
            else:
                # print(os.path.join(self.input_isg_dir, str(self.info['images'][ix_isg]['id']) + '.npy'))
                _isg = np.load(os.path.join(self.input_isg_dir, str(self.info_isg['images'][ix_isg]['id']) + '.npy'))
                for k in range(seq_per_img): isg_batch[i * seq_per_img + k, :] = _isg
                _ssg = np.load(os.path.join(self.input_ssg_dir, str(self.info['images'][ix_ssg]['id']) + '.npy'))
                for k in range(seq_per_img): ssg_batch[i * seq_per_img + k, :] = _ssg

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['id_isg'] = self.info_isg['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        data = {}

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images

        data['isg_feats'] = isg_batch # if pre_ft is 0, then it equals None
        data['ssg_feats'] = ssg_batch # if pre_ft is 0, then it equals None
        data['labels'] = np.vstack(label_batch)
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos
        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        if self.opt.train_split == 'train':
            ix_isg = self.isg_unorder[ix]
            ix_ssg = self.ssg_unorder[ix]
        else:
            ix_isg = ix
            ix_ssg = ix
        return (np.load(os.path.join(self.input_isg_dir, str(self.info['images'][ix_isg]['id']) + '.npy')),
                np.load(os.path.join(self.input_ssg_dir, str(self.info['images'][ix_ssg]['id']) + '.npy')),
                ix)

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0,  # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]