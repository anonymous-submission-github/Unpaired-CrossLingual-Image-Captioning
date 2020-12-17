# coding=utf-8

"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
# import skimage.io
from PIL import Image
import pdb
import re
import jieba
import io
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# jieba.enable_paddle()

pattern = re.compile(
    u"[–0123456789/~*'зД=つ\-Дノ→\+·②⑥:°【［•”˙×ε○↖\\、↗∧⊙⊥⑨┌╯▽罒】☆⌒∩〃￥〕ㄒㄥ〔`\[\"\?\]ò\*ω￥∀Σπ●《》……^≦～－——？()\.／：；‘“、）（……％?¥＃/#@$%^&£]")
splitPattern = re.compile(u"[，,!！~。;]")


def cleanData(toclean):
    rt = toclean.replace(u"?", "")
    rt = re.sub(pattern, "", rt)
    rt = re.sub(splitPattern, " ", rt)
    return rt


def cleanData2(toclean):
  rt = toclean.replace(u"?", "")
  rt = re.sub(pattern, "", rt)
  rt = re.sub(splitPattern, " ", rt)
  return list(filter(lambda x: len(x) > 0, rt.split(" ")))


def readData(path):
  li = []
  with open(path, "r") as f:
    for i in f:
      li.append(i.strip())
  return li

def tokenize(line,delim=' '):
    # replace non-breaking whitespace
    _line = line.replace("\xa0", " ").strip()
    # tokenize
    _tok = jieba.lcut(_line.rstrip('\r\n'),cut_all=False)
    # delete " " in tok list
    _tok_new=[]
    for word in _tok:
      if word!=" ":
        _tok_new.append(word)
    # _tokenized = delim.join(_tok)
    return _tok_new


def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  counts = {}
  for img in imgs:
    for sent in img['sentences']:
      for w in sent['tokens']:
        counts[w] = counts.get(w, 0) + 1

  with io.open(params['output_json']+'dict_counts.json', 'w', encoding="utf-8") as outfile:
    outfile.write(json.dumps(counts, ensure_ascii=False))

  cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20])))

  # print some stats
  total_words = sum(counts.values())
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr]
  vocab = [w for w,n in counts.items() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  print('number of words in vocab would be %d' % (len(vocab), ))
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for img in imgs:
    for sent in img['sentences']:
      txt = sent['tokens']
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  for i in range(max_len+1):
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
  for img in imgs:
    img['final_captions'] = []
    for sent in img['sentences']:
      txt = sent['tokens']
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      if len(caption) == 0:
        print(img['sentences'])
        # pdb.set_trace()
      img['final_captions'].append(caption)

  return vocab, counts

def encode_captions(imgs, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      if min(max_length, len(s)) == 0:
        print(s)
        # pdb.set_trace()
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      tmp_s = ''
      for k,w in enumerate(s):
        if k < max_length:
          Li[j, k] = wtoi[w]
          tmp_s = tmp_s + w + ' '

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  # assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length

def main_google(params):
#
## read json
  imgs = json.load(open(params['input_path'], 'r'))
  key_list=imgs.keys()
  import random
  random.seed(7)
  random.shuffle(key_list)

  # use coco-cn split 1000 as test data
  cocobu_cococn=json.load(open('data/coco_cn/cocobu_gan_isg.json'))
  cocobu_test_ids=[img['id'] for img in cocobu_cococn['images'] if img['split']=='test']
  key_list_coco=cocobu_test_ids
  key_list_aicv6 = [i for i in key_list if 'coco' not in i]

  val_ix=key_list_aicv6[:10000]
  test_ix=key_list_coco
  train_ix=key_list_aicv6[10000:]

  img_ids_save=train_ix + val_ix +test_ix

  imgs_new=[]
  count=0
  for i in img_ids_save:
  # for i in imgs.keys():
    tmp={}
    # li=imgs[i].split(' ')
    # when only one caption
    temp = cleanData(imgs[i].decode('utf-8'))
    # when caption stored in list
    # temp = cleanData(imgs[i][0].decode('utf-8'))
    li=tokenize(temp)
    tmp['sentids']=[]
    tmp['sentids'].append(i)
    tmp['sentences']=[]
    sub_sent={}
    sub_sent['tokens']=li
    sub_sent['raw']=temp
    tmp['sentences'].append(sub_sent)
    if i in train_ix:
      tmp['split']='train'
    elif i in val_ix :
      tmp['split']='val'
    elif i in test_ix:
      tmp['split']='test'
    tmp['sen_id']=i
    imgs_new.append(tmp)
    count+=1
    if count%10000==0:
      print ('{} sentences loaded...'.format(count))
      # break

  imgs =imgs_new

  seed(123) # make reproducible
  coco_is_dict=np.load('data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz',allow_pickle=True)['spice_dict']

  # create the vocab

  vocab, counts = build_vocab(imgs, params)
  # itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  # wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
  itow = coco_is_dict[()]['ix_to_word']
  wtoi = coco_is_dict[()]['word_to_ix']

  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

  # create output h5 file
  N = len(imgs)
  f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.close()

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['counts'] = counts
  out['images'] = []
  for i,img in enumerate(imgs):

    jimg = {}
    jimg['split'] = img['split']
    jimg['id']=img['sen_id']
    jimg['file_path']=img['split']+"_"+str(i)

    # if 'cocoid' in img: jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)
    out['images'].append(jimg)  ## sentence_id and split actually.


  # json.dump(out, io.open(params['output_json'], 'w', encoding="utf-8"),ensure_ascii=False)
  with io.open(params['output_json'], 'w', encoding="utf-8") as outfile:
    outfile.write(json.dumps(out, ensure_ascii=False))
  print('wrote ', params['output_json'])


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_path', default='data/aic_process/ALL_11683_COCOCN_zh_v3.json', help='input wmt data file to process into hdf5')
  parser.add_argument('--output_json', default='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5.json', help='output json file')
  parser.add_argument('--output_h5', default='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5', help='output h5 file')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main_google(params)
