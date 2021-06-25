import os
import json
import numpy as np
import torch
from collections import defaultdict 

def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def invert_dict(d):
  return {v: k for k, v in d.items()}
  

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def load_scenes(scenes_json):
    try:
        with open(scenes_json) as f:
            scenes_dict = json.load(f)['scenes']
    except:
        with open(scenes_json, encoding='utf-8') as f:
            scenes_dict = json.load(f)
            print(scenes_dict.keys())
        
    scenes = defaultdict(list)
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            item['2d_coords'] = o['2d_coords']
            item['category'] = o['category']
            table.append(item)
        scenes[s['image_index']] = table
    return scenes
    

def load_embedding(path):
    return torch.Tensor(np.load(path))