import os
import json
import random
from collections import Counter
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


# data utils
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


def read_token_clf_data(f_path, sep='\t'):
    sentences, labels = [], []
    tokens, token_labels = [], []
    with open(f_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split(sep)
                tokens.append(token)
                token_labels.append(label)
            else:
                assert len(tokens) == len(token_labels)
                sentences.append(tokens)
                labels.append(token_labels)
                tokens = []
                token_labels = []
    return sentences, labels


def create_token_clf_examples(sentences, labels):
    examples = []
    for tokens, token_labels in zip(sentences, labels):
        examples.append(TokenClfExample(tokens, token_labels))
    return examples


class TokenClfExample():
    def __init__(self, tokens, token_labels):
        self.tokens = tokens
        self.token_labels = token_labels


class TokenClfFeature():
    def __init__(self, token_ids, token_label_ids):
        self.token_ids = token_ids
        self.token_label_ids = token_label_ids


class MsraNerProcessor():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        sentences, labels = read_token_clf_data(
            os.path.join(self.data_dir, 'train.txt'))
        train_examples = create_token_clf_examples(sentences, labels)
        return train_examples
        pass

    def get_test_examples(self):
        sentences, labels = read_token_clf_data(
            os.path.join(self.data_dir, 'test.txt'))
        test_examples = create_token_clf_examples(sentences, labels)
        return test_examples
        pass

    @classmethod
    def get_label2id(cls):
        label2id = {'_pad': 0, 'O': 1, 'B-PER': 2, 'I-PER': 3,
                    'B-LOC': 4, 'I-LOC': 5, 'B-ORG': 6, 'I-ORG': 7}
        return label2id


def examples_to_features(examples, tokenizer, label2id, max_len=64):
    features = []
    lens = []
    num_examples = len(examples)
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            logger.info('Writting examples {} of {}'.format(idx, num_examples))
        tokens = example.tokens
        token_labels = example.token_labels
        token_ids = tokenizer.tokens_to_ids(tokens)
        token_label_ids = [label2id[label] for label in token_labels]
        lens.append(len(token_ids))

        token_ids = token_ids[:max_len]
        token_label_ids = token_label_ids[:max_len]
        features.append(TokenClfFeature(token_ids, token_label_ids))
        if idx < 3:
            logger.info('*** Example ***')
            logger.info('tokens: {}'.format(str(tokens)))
            logger.info('token_ids: {}'.format(str(token_ids)))
            logger.info('token_labels: {}'.format(str(token_labels)))
            logger.info('token_label_ids: {}'.format(str(token_label_ids)))
    logger.info('lens: mean: {}, std: {}, max: {}, min: {}'.format(
        np.mean(lens), np.std(lens), max(lens), min(lens)))
    return features


class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)


class TokenClfCollate():
    def __init__(self, predict=False):
        self.predict = predict

    def __call__(self, batch):
        max_len = max([len(f.token_ids) for f in batch])
        for idx, feature in enumerate(batch):
            padding = [0] * (max_len - len(feature.token_ids))
            batch[idx].token_ids += padding
            batch[idx].token_label_ids += padding
        token_ids = torch.LongTensor([f.token_ids for f in batch])
        if self.predict:
            return token_ids
        token_label_ids = torch.LongTensor([f.token_label_ids for f in batch])
        return token_ids, token_label_ids


# tokenizer utils
class CharTokenizer():
    def __init__(self, model_dir):
        f_path = os.path.join(model_dir, 'vocab.json')
        with open(f_path, 'r', encoding='utf8') as f:
            self.vocab = json.load(f)

    def tokenize(self, text):
        tokens = [char.strip() for char in text if char.strip()]
        return tokens

    def tokens_to_ids(self, tokens, _unk_idx=1):
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, _unk_idx))
        return ids

    @classmethod
    def build_vocab(cls, sentences, output_dir, min_count=1, _pad_idx=0, _unk_idx=1, preprocess_fn=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vocab = {'_pad': _pad_idx, '_unk': _unk_idx}
        token_counter = Counter()
        for tokens in sentences:
            token_counter.update(tokens)
        for token, count in token_counter.items():
            if count < min_count:
                continue
            if token not in vocab:
                vocab[token] = len(vocab)
        vocab_path = os.path.join(output_dir, 'vocab.json')
        save_dict_to_json_file({t: c for t, c in token_counter.most_common()},
                               vocab_path.replace('vocab', 'token_count'))
        save_dict_to_json_file(vocab, vocab_path)
        logger.info('Building vocabulary of {} tokens to {}'.format(
            len(token_counter), len(vocab)))


# experiments utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config():
    def __init__(self, model_dir):
        f_path = os.path.join(model_dir, 'config.json')
        logger.info('Loading config from: {}'.format(f_path))
        with open(f_path, 'r', encoding='utf8') as f:
            config = json.load(f)
        self.__dict__.update(config)
        logger.info('Config: {}'.format(self.__dict__))

    def save(self, output_dir):
        f_path = os.path.join(output_dir, 'config.json')
        with open(f_path, 'w', encoding='utf8') as f:
            json.dump(self.__dict__, f, indent=4)
        logger.info('Saving config to: {}'.format(f_path))


# metrics utils
def acc_p_r_f1(preds, labels, mask):
    """
        preds: (batch_size*seq_len)
        labels: (batch_size*seq_len)
        mask: (batch_size*seq_len)
    """
    active_preds = preds[mask]
    active_labels = labels[mask]
    assert active_preds.shape == active_labels.shape
    acc = accuracy_score(active_labels, active_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        active_labels, active_preds, average='micro')
    return {'acc': acc, 'p': p, 'r': r, 'f1': f1}


if __name__ == "__main__":
    train_sentences, train_labels = read_token_clf_data('data/msra/train.txt')
    test_sentences, test_labels = read_token_clf_data('data/msra/test.txt')
    all_sentences = train_sentences + test_sentences
    CharTokenizer.build_vocab(all_sentences, output_dir='experiments/bilstm')
    pass
