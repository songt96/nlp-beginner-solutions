import os
import re
import csv
import json
import six
import time
import logging
from collections import Counter
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
import random
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


# preprocess utils
def quan2ban(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换为半角空格
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += six.unichr(inside_code)
    return rstring


RE_MULTI_BLANK = re.compile(r'\s{2,}')
MULTI_BLANK_SUB = ' '
RE_HTML = re.compile(r'</?[a-z]{1,2}>')  # 89 html tag
HTML_SUB = ''
RE1 = re.compile(r'[-]{2,}')  # 268
RE1_SUB = '-'
RE2 = re.compile(r'[_]{2,}')  # 79
RE2_SUB = '_'
RE3 = re.compile(r'\*{2,}')  # 139
RE3_SUB = '*'
RE4 = re.compile(r'[^\u4e00-\u9fa5\w ]')
RE4_SUB = ''
RE_PUNC = re.compile(r'[,;\.\?!:\'"，；。？！：‘’“”·、\(\)（）《》【】]')
PUNC_SUB = ''


def preprocess_lcqmc(text, lower=True, q2b=True, clean=True):
    text = text.strip()
    if lower:
        text = text.lower()
    if q2b:
        text = quan2ban(text)
    if clean:
        text = RE_MULTI_BLANK.sub(MULTI_BLANK_SUB, text)
        text = RE_HTML.sub(HTML_SUB, text)
        text = RE1.sub(RE1_SUB, text)
        text = RE2.sub(RE2_SUB, text)
        text = RE3.sub(RE3_SUB, text)
        # text = RE_PUNC.sub(PUNC_SUB, text)
    text = text.strip()
    return text


# data utils
def read_csv(file_path, quotechar='"', delimiter=','):
    """Reads a `,` or `\t` separated value file."""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


class TextPairExample():
    def __init__(self, text_a, text_b, label=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_lcqmc_examples(lines):
    examples = []
    for line in lines:
        assert len(line) == 3
        text_a = line[0]
        text_b = line[1]
        label = line[2]
        examples.append(TextPairExample(text_a, text_b, label))
    return examples


class LcqmcProcessor():
    def __init__(self, data_dir):
        self.train_examples_path = os.path.join(data_dir, 'train.tsv')
        self.dev_examples_path = os.path.join(data_dir, 'dev.tsv')
        self.test_examples_path = os.path.join(data_dir, 'test.tsv')

    def get_train_examples(self):
        return create_lcqmc_examples(read_csv(self.train_examples_path, delimiter='\t'))

    def get_dev_examples(self):
        return create_lcqmc_examples(read_csv(self.dev_examples_path, delimiter='\t'))

    def get_test_examples(self):
        return create_lcqmc_examples(read_csv(self.test_examples_path, delimiter='\t'))

    @classmethod
    def get_labels(cls):
        labels = ['0', '1']
        return labels

    @classmethod
    def get_label2id(cls):
        label2id = {'0': 0, '1': 1}
        return label2id


class TextPairFeature():
    def __init__(self, ids_a, ids_b, label_id):
        self.ids_a = ids_a
        self.ids_b = ids_b
        self.label_id = label_id


def examples_to_features(examples, label2id, tokenizer, max_len=48,
                         preprocess_fn=preprocess_lcqmc, verbose=True):

    features = []
    len_a, len_b = [], []
    num_examples = len(examples)
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            logger.info('Writing examples {} of {}'.format(idx, num_examples))
        text_a = example.text_a
        text_b = example.text_b
        if preprocess_fn is not None:
            text_a = preprocess_lcqmc(text_a)
            text_b = preprocess_lcqmc(text_b)
        tokens_a = tokenizer.tokenize(text_a)
        len_a.append(len(tokens_a))
        tokens_b = tokenizer.tokenize(text_b)
        len_b.append(len(tokens_b))

        ids_a = tokenizer.tokens_to_ids(tokens_a)
        ids_b = tokenizer.tokens_to_ids(tokens_b)

        ids_a = ids_a[:max_len]
        ids_b = ids_b[:max_len]
        label_id = label2id.get(example.label)
        if idx < 3 and verbose:
            logger.info('*** Example ***')
            logger.info('tokens_a: {}'.format(str(tokens_a)))
            logger.info('ids_a: {}'.format(str(ids_a)))
            logger.info('tokens_b: {}'.format(str(tokens_b)))
            logger.info('ids_b: {}'.format(str(ids_b)))
            logger.info('label: {} (id = {})'.format(
                example.label, label_id))
        features.append(TextPairFeature(ids_a, ids_b, label_id))
    logger.info('A: mean: {}, std: {}, max: {}, min: {}'.format(
        np.mean(len_a), np.std(len_a), max(len_a), min(len_a)))
    logger.info('B: mean: {}, std: {}, max: {}, min: {}\n'.format(
        np.mean(len_b), np.std(len_b), max(len_b), min(len_b)))
    return features


class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)


class TextPairCollate():
    def __init__(self, predict=False,):
        self.predict = predict

    def __call__(self, batch):
        max_len_a = max([len(feature.ids_a) for feature in batch])
        max_len_b = max([len(feature.ids_b) for feature in batch])
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


# tokenizer utils
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


class Tokenizer():
    def save_vocab(self, model_dir):
        save_dict_to_json_file(
            self.vocab, os.path.join(model_dir, 'vocab.json'))


class CharTokenizer(Tokenizer):
    def __init__(self, model_dir, ):
        super(CharTokenizer, self).__init__()
        self.vocab = load_dict_from_json_file(
            os.path.join(model_dir, 'vocab.json'))
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        tokens = [char.strip() for char in text if char.strip()]
        return tokens

    def tokens_to_ids(self, tokens, _unk_idx=1):
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, _unk_idx))
        return ids

    @classmethod
    def build_vocab(cls, texts, output_dir, min_count=1, preprocess_fn=preprocess_lcqmc, _pad_idx=0, _unk_idx=1,):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vocab = {'_pad': _pad_idx, '_unk': _unk_idx}
        token_counter = Counter()
        for text in texts:
            if preprocess_fn is not None:
                text = preprocess_fn(text)
            token_counter.update(text)
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


if __name__ == "__main__":
    data_dir = './data/lcqmc'
    processor = LcqmcProcessor(data_dir)
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()
    test_examples = processor.get_test_examples()
    all_examples = train_examples + dev_examples + test_examples
    all_texts = [example.text_a for example in all_examples] + \
        [example.text_b for example in all_examples]
    CharTokenizer.build_vocab(
        all_texts, output_dir='experiments/esim_char', min_count=1)

    pass
