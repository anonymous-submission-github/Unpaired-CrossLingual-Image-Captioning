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
import misc.utils as utils


import multiprocessing


class DataLoader_UP(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
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
        self.use_att = getattr(opt, 'use_att', False)
        self.use_ssg = getattr(opt, 'use_ssg', 0)
        self.use_isg = getattr(opt, 'use_isg', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))

        # load the vocab info from info_path (en coco)
        # self.ix_to_word = opt.vocab
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)

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
        try:
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        except:
            self.h5_label_file = h5py.File(str(''+self.opt.input_label_h5), 'r', driver='core')


        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

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
        self.iterators = {'train': 0, 'val': 0, 'test': 0, 'train_sg': 0, 'val_sg': 0, 'test_sg': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

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
            # print (seq)

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        isg_rela_batch = []
        isg_obj_batch = []
        isg_attr_batch = []

        ssg_rela_batch = []
        ssg_obj_batch = []
        ssg_attr_batch = []

        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_isg, tmp_ssg, ix, tmp_wrapped = self._prefetch_process[split].get()
            # print('batch'+str(i))
            # print ('fc:')
            # print(tmp_fc.shape)
            # print(tmp_fc)
            # print ('att:')
            # print(tmp_att)
            # print ('isg:')
            # print (tmp_isg)
            # print ('ssg:')
            # print(tmp_ssg)
            # print ('wrapped:')
            # print (tmp_wrapped)
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            isg_rela_batch.append(tmp_isg['ssg_rela_matrix'])
            isg_attr_batch.append(tmp_isg['ssg_attr'])
            isg_obj_batch.append(tmp_isg['ssg_obj'])

            ssg_rela_batch.append(tmp_ssg['ssg_rela_matrix'])
            ssg_attr_batch.append(tmp_ssg['ssg_attr'])
            ssg_obj_batch.append(tmp_ssg['ssg_obj'])

            # print('ssg_o')
            # print(tmp_ssg['ssg_obj'])
            # print('ssg_a')
            # print(tmp_ssg['ssg_attr'])
            # print('ssg_r')
            # print(tmp_ssg['ssg_rela_matrix'])

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            # info_dict['file_path'] = ''
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        fc_batch, att_batch, label_batch, gts, infos = zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: 0,reverse=True))
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] * seq_per_img for _ in fc_batch]))
        max_att_len = max([_.shape[0] for _ in att_batch])

        # merge att_feats
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        # if data['att_masks'].sum() == data['att_masks'].size:
        #     data['att_masks'] = None

        if self.use_isg:
            max_rela_len = max([_.shape[0] for _ in isg_rela_batch])
            data['isg_rela_matrix'] = np.ones([len(att_batch) * seq_per_img, max_rela_len, 3]) * -1
            for i in range(len(isg_rela_batch)):
                data['isg_rela_matrix'][i * seq_per_img:(i + 1) * seq_per_img, 0:len(isg_rela_batch[i]), :] = \
                    isg_rela_batch[i]
            data['isg_rela_masks'] = np.zeros(data['isg_rela_matrix'].shape[:2], dtype='float32')
            for i in range(len(isg_rela_batch)):
                data['isg_rela_masks'][i * seq_per_img:(i + 1) * seq_per_img, :isg_rela_batch[i].shape[0]] = 1

            max_obj_len = max([_.shape[0] for _ in isg_obj_batch])
            data['isg_obj'] = np.ones([len(att_batch) * seq_per_img, max_obj_len]) * -1
            for i in range(len(isg_obj_batch)):
                data['isg_obj'][i * seq_per_img:(i + 1) * seq_per_img, 0:len(isg_obj_batch[i])] = isg_obj_batch[i]
            data['isg_obj_masks'] = np.zeros(data['isg_obj'].shape, dtype='float32')
            for i in range(len(isg_obj_batch)):
                data['isg_obj_masks'][i * seq_per_img:(i + 1) * seq_per_img, :isg_obj_batch[i].shape[0]] = 1

            max_attr_len = max([_.shape[1] for _ in isg_attr_batch])
            data['isg_attr'] = np.ones([len(att_batch) * seq_per_img, max_obj_len, max_attr_len]) * -1
            for i in range(len(isg_obj_batch)):
                data['isg_attr'][i * seq_per_img:(i + 1) * seq_per_img, 0:len(isg_obj_batch[i]),
                0:isg_attr_batch[i].shape[1]] = \
                    isg_attr_batch[i]
            data['isg_attr_masks'] = np.zeros(data['isg_attr'].shape, dtype='float32')
            for i in range(len(isg_attr_batch)):
                for j in range(len(isg_attr_batch[i])):
                    N_attr_temp = np.sum(isg_attr_batch[i][j, :] >= 0)
                    data['isg_attr_masks'][i * seq_per_img: (i + 1) * seq_per_img, j, 0:int(N_attr_temp)] = 1

        if self.use_ssg:
            max_rela_len = max([_.shape[0] for _ in ssg_rela_batch])
            data['ssg_rela_matrix'] = np.ones([len(att_batch) * seq_per_img, max_rela_len, 3]) * -1
            for i in range(len(ssg_rela_batch)):
                data['ssg_rela_matrix'][i * seq_per_img:(i + 1) * seq_per_img, 0:len(ssg_rela_batch[i]), :] = ssg_rela_batch[i]
            data['ssg_rela_masks'] = np.zeros(data['ssg_rela_matrix'].shape[:2], dtype='float32')
            for i in range(len(ssg_rela_batch)):
                data['ssg_rela_masks'][i * seq_per_img:(i + 1) * seq_per_img, :ssg_rela_batch[i].shape[0]] = 1

            max_obj_len = max([_.shape[0] for _ in ssg_obj_batch])
            data['ssg_obj'] = np.ones([len(att_batch) * seq_per_img, max_obj_len]) * -1
            for i in range(len(ssg_obj_batch)):
                data['ssg_obj'][i * seq_per_img:(i + 1) * seq_per_img, 0:len(ssg_obj_batch[i])] = ssg_obj_batch[i]
            data['ssg_obj_masks'] = np.zeros(data['ssg_obj'].shape, dtype='float32')
            for i in range(len(ssg_obj_batch)):
                data['ssg_obj_masks'][i * seq_per_img:(i + 1) * seq_per_img, :ssg_obj_batch[i].shape[0]] = 1

            max_attr_len = max([_.shape[1] for _ in ssg_attr_batch])
            data['ssg_attr'] = np.ones([len(att_batch) * seq_per_img, max_obj_len, max_attr_len]) * -1
            for i in range(len(ssg_obj_batch)):
                data['ssg_attr'][i * seq_per_img:(i + 1) * seq_per_img, 0:len(ssg_obj_batch[i]),0:ssg_attr_batch[i].shape[1]] = ssg_attr_batch[i]
            data['ssg_attr_masks'] = np.zeros(data['ssg_attr'].shape, dtype='float32')
            for i in range(len(ssg_attr_batch)):
                for j in range(len(ssg_attr_batch[i])):
                    N_attr_temp = np.sum(ssg_attr_batch[i][j, :] >= 0)
                    data['ssg_attr_masks'][i * seq_per_img: (i + 1) * seq_per_img, j, 0:int(N_attr_temp)] = 1

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos


        # print('ssg_o_m')
        # print(data['ssg_obj_masks'])
        # print('ssg_a_m')
        # print(data['ssg_attr_masks'])
        # print('ssg_r_m')
        # print(data['ssg_rela_masks'])
        # sents = utils.decode_sequence(self.get_vocab(), data['labels'] )
        # print(sents)

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        ssg_data = {}
        ssg_data['ssg_rela_matrix'] = {}
        ssg_data['ssg_attr'] = {}
        ssg_data['ssg_obj'] = {}
        isg_data = {}
        isg_data['ssg_rela_matrix'] = {}
        isg_data['ssg_attr'] = {}
        isg_data['ssg_obj'] = {}
        if self.use_att:
            # att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            # Reshape to K x C
            # att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            att_feat = np.zeros((50,2048),dtype='float32')
            fc_feat=np.zeros(2048,dtype='float32')

            if self.use_isg:
                path_temp = os.path.join(self.input_isg_dir, str(self.info['images'][ix]['id']) + '.npy')
                if os.path.isfile(path_temp):
                    isg_info = np.load(os.path.join(path_temp))
                    isg_rela_matrix = isg_info[()]['rela_info']
                    isg_obj_att_info = isg_info[()]['obj_info']

                    len_obj = len(isg_obj_att_info)
                    isg_obj = np.zeros([len_obj, ])
                    if len_obj == 0:
                        isg_rela_matrix = np.zeros([0, 3])
                        isg_attr = np.zeros([0, 1])
                        isg_obj = np.zeros([0, ])
                    else:
                        max_attr_len = max([len(_) for _ in isg_obj_att_info])
                        isg_attr = np.ones([len_obj, max_attr_len - 1]) * -1
                        for i in range(len_obj):
                            isg_obj[i] = isg_obj_att_info[i][0]
                            for j in range(1, len(isg_obj_att_info[i])):
                                isg_attr[i, j - 1] = isg_obj_att_info[i][j]

                    isg_data = {}
                    isg_data['ssg_rela_matrix'] = isg_rela_matrix
                    isg_data['ssg_attr'] = isg_attr
                    isg_data['ssg_obj'] = isg_obj
                else:
                    isg_data = {}
                    isg_data['ssg_rela_matrix'] = np.zeros([0, 3])
                    isg_data['ssg_attr'] = np.zeros([0, 1])
                    isg_data['ssg_obj'] = np.zeros([0, ])
            if self.use_ssg:
                # print('ix: {}'.format(ix))
                # print(self.info['images'][ix])
                # print('id: {}'.format(self.info['images'][ix]['id']))
                path_temp = os.path.join(self.input_ssg_dir, str(self.info['images'][ix]['id']) + '.npy')
                # try:
                #     path_temp = os.path.join(self.input_ssg_dir, str(self.info['images'][ix]['id']) + '.npy')
                # except:
                #     print('the loss ix: {}'.format(ix))

                if os.path.isfile(path_temp):
                    ssg_info = np.load(os.path.join(path_temp),allow_pickle=True)
                    ssg_rela_matrix = ssg_info[()]['rela_info']
                    ssg_obj_att_info = ssg_info[()]['obj_info']

                    len_obj = len(ssg_obj_att_info)
                    ssg_obj = np.zeros([len_obj, ])
                    if len_obj == 0:
                        ssg_rela_matrix = np.zeros([0, 3])
                        ssg_attr = np.zeros([0, 1])
                        ssg_obj = np.zeros([0, ])
                    else:
                        max_attr_len = max([len(_) for _ in ssg_obj_att_info])
                        ssg_attr = np.ones([len_obj, max_attr_len - 1]) * -1
                        for i in range(len_obj):
                            ssg_obj[i] = ssg_obj_att_info[i][0]
                            for j in range(1, len(ssg_obj_att_info[i])):
                                ssg_attr[i, j - 1] = ssg_obj_att_info[i][j]

                    ssg_data = {}
                    ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                    ssg_data['ssg_attr'] = ssg_attr
                    ssg_data['ssg_obj'] = ssg_obj
                else:
                    ssg_data = {}
                    ssg_data['ssg_rela_matrix'] = np.zeros([0, 3])
                    ssg_data['ssg_attr'] = np.zeros([0, 1])
                    ssg_data['ssg_obj'] = np.zeros([0, ])
        else:
            att_feat = np.zeros(((1, 1, 1)))

        return (np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id'][5:]) + '.npy'),allow_pickle=True),
        # return (fc_feat,
                att_feat,
                isg_data,
                ssg_data,
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
        # print ('ri:'+str(ri))
        # print ('ix:'+str(ix))
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

        assert tmp[4] == ix, "ix not equal"
        return tmp + [wrapped]