import json
import os
import re
import six
import time
import logging
from collections import Counter
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
import jieba
import shutil
import random
import numpy as np
jieba.initialize()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


# data utils
class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)


class TextPairCollate():
    def __init__(self, max_len=None, predict=False,):
        self.max_len = max_len
        self.predict = predict

    def __call__(self, batch):
        len_a = [len(feature.ids_a) for feature in batch]
        len_b = [len(feature.ids_b) for feature in batch]
        max_len_a = max(len_a)
        max_len_b = max(len_b)
        if self.max_len is not None:
            assert self.max_len >= max_len_a and self.max_len >= max_len_b
        for idx, feature in enumerate(batch):
            padding_a = [0] * (max_len_a - len(feature.ids_a))  # len_a[idx]
            padding_b = [0] * (max_len_b - len(feature.ids_b))  # len_b[idx]
            batch[idx].ids_a += padding_a
            batch[idx].ids_b += padding_b
            assert len(batch[idx].ids_a) == max_len_a
            assert len(batch[idx].ids_b) == max_len_b
        ids_a = torch.LongTensor([f.ids_a for f in batch])
        ids_b = torch.LongTensor([f.ids_b for f in batch])
        if self.predict:
            return ids_a, ids_b
        label_id = torch.LongTensor([f.label_id for f in batch])
        return ids_a, ids_b, label_id


def load_embedding_dict(embedding_path, embedding_dim):
    embedding_dict = {}
    start = time.time()
    with open(embedding_path, 'r', encoding='utf8') as f:
        for line in f:
            kv = line.rsplit(None, embedding_dim)
            if len(kv) != embedding_dim + 1:
                continue
            token = kv[0]
            embedding = np.asarray(kv[1:], dtype=np.float)
            embedding_dict[token] = embedding
    logger.info('Loaded {} embeddings from {} cost {:.1f}s'.format(
        len(embedding_dict), embedding_path, time.time() - start))
    return embedding_dict


def get_embedding(vocab, embedding_path, embedding_dim, _pad_idx=0):
    vocab_size = len(vocab)
    covered_embedding_path = './embedding/covered_embedding_V{}_D{}'.format(
        vocab_size, embedding_dim)
    exists_covered_embedding = os.path.exists(covered_embedding_path)
    if exists_covered_embedding:
        embedding_dict = load_embedding_dict(
            covered_embedding_path, embedding_dim)
    else:
        embedding_dict = load_embedding_dict(embedding_path, embedding_dim)
    n_covered = 0
    embeddings = np.random.randn(vocab_size, embedding_dim)  # * 0.1
    embeddings[_pad_idx] = 0
    covered_embedding_dict = {}
    for token, idx in vocab.items():
        if token in embedding_dict:
            embedding = embedding_dict.get(token)
            embeddings[idx] = embedding
            n_covered += 1
            if not exists_covered_embedding:
                covered_embedding_dict[token] = embedding
    logger.info('Vocab size: {}, embeddings covered: {} tokens'.format(
        vocab_size, n_covered))
    if not exists_covered_embedding:
        with open(covered_embedding_path, 'w', encoding='utf8') as f:
            for token, embedding in covered_embedding_dict.items():
                f.write('{} {}\n'.format(token, ' '.join(map(str, embedding))))
        logger.info('Saved covered embeddings to {}'.format(
            covered_embedding_path))
    return embeddings


# experiments utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dict_from_json_file(f_path):
    with open(f_path, 'r', encoding='utf8') as f:
        dic = json.load(f)
        logger.info('Loaded dict of {} items from {}'.format(
            len(dic), f_path))
        return dic


def save_dict_to_json_file(dic, f_path):
    with open(f_path, 'w', encoding='utf8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)
    logger.info('Saving dict of {} items to {}'.format(len(dic), f_path))


class Config:
    def __init__(self, model_dir):
        f_path = os.path.join(model_dir, 'config.json')
        logger.info('Loading config from: {}'.format(f_path))
        with open(f_path) as f:
            config = json.load(f)
            self.__dict__.update(config)
        logger.info('Config: {}'.format(config))

    def save(self, model_dir):
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        """Gives dict-like access to Config instance by `config.dict['learning_rate']"""
        return self.__dict__


def save_experiment(config, model, val_result, output_dir, test_result=None):
    logger.info('\nSaving experiment')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config.save(output_dir)
    logger.info('Best val result: {}'.format(json.dumps(val_result)))
    # Save model
    torch.save(model.state_dict(), os.path.join(
        output_dir, 'pytorch_model.bin'))

    # Save val result
    save_dict_to_json_file(val_result, os.path.join(
        output_dir, 'best_val_result.json'))
    # Save test result
    if test_result is not None:
        logger.info('Best test result: ', test_result)
        save_dict_to_json_file(test_result, os.path.join(
            output_dir, 'test_result.json'))


# metrics utils
def acc_p_r_f1(preds, labels):
    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'acc': acc,
        'p': p,
        'r': r,
        'f1': f1
    }
