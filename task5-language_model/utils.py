import os
import json
import random
import logging
from collections import Counter
import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


# data utils
def read_lm_examples(f_path):
    examples = []
    with open(f_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(line)
    return examples


class LmProcessor():
    def __init__(self, data_dir):
        self.train_examples_path = os.path.join(data_dir, 'train.txt')
        self.dev_examples_path = os.path.join(data_dir, 'valid.txt')
        self.test_examples_path = os.path.join(data_dir, 'test.txt')

    def get_train_examples(self):
        return read_lm_examples(self.train_examples_path)

    def get_dev_examples(self):
        return read_lm_examples(self.dev_examples_path)

    def get_test_examples(self):
        return read_lm_examples(self.test_examples_path)


def examples_to_ids(examples, tokenizer, max_len=None, verbose=True):
    all_ids = []
    lens = []
    num_examples = len(examples)
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            print('Writing examples {} of {}: '.format(idx, num_examples))
        text = example.lower()
        tokens = tokenizer.tokenize(text)
        tokens += ['<eos>']
        lens.append(len(tokens))
        ids = tokenizer.tokens_to_ids(tokens)
        ids = ids[:max_len]
        if idx < 3 and verbose:
            logger.info('*** Example ***')
            logger.info('tokens: {}'.format(str(tokens)))
            logger.info('ids: '.format(str(ids)))
        all_ids.extend(ids)
    logger.info('mean: {}, std: {}, min: {}, max: {}'.format(
        np.mean(lens), np.std(lens), min(lens), max(lens)))
    return all_ids


class LMDataLoader():
    def __init__(self, ids, batch_size=64, max_len=35, shuffle=False):
        self.batch_size = batch_size
        self.max_len = max_len
        num_tokens = len(ids)
        self.num_batches = num_tokens // batch_size
        tensor = torch.LongTensor(
            ids[:self.num_batches * batch_size]).view(batch_size, self.num_batches)
        if shuffle:
            self.tensor = tensor[torch.randperm(batch_size)]
        else:
            self.tensor = tensor

    def __len__(self):
        return (self.num_batches + self.max_len - 1) // self.max_len

    def __iter__(self):
        for i in range(0, self.num_batches, self.max_len):
            end = min(i + self.max_len, self.num_batches - 1)
            inputs = self.tensor[:, i: end]
            targets = self.tensor[:, i+1: end+1]
            yield inputs, targets


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


# tokenizer utils
class Tokenizer():
    def save_vocab(self, output_dir):
        save_dict_to_json_file(
            self.vocab, os.path.join(output_dir, 'vocab.json'))


class WhitespaceTokenizer(Tokenizer):
    def __init__(self, model_dir):
        super(WhitespaceTokenizer, self).__init__()
        self.vocab = load_dict_from_json_file(
            os.path.join(model_dir, 'vocab.json'))

    def tokenize(self, text):
        tokens = [token.strip()
                  for token in text.strip().split() if token.strip()]
        return tokens

    def tokens_to_ids(self, tokens):
        ids = []
        unk_idx = self.vocab.get('<unk>')
        for token in tokens:
            ids.append(self.vocab.get(token, unk_idx))
        return ids

    @classmethod
    def build_vocab(cls, texts, output_dir, min_count=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        texts = list(set(texts))
        vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        token_counter = Counter()
        for text in texts:
            text = text.strip().lower()
            tokens = [token.strip() for token in text.split() if token.strip()]
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


class CharTokenizer(Tokenizer):
    def __init__(self, model_dir):
        super(CharTokenizer, self).__init__()
        self.vocab = load_dict_from_json_file(
            os.path.join(model_dir, 'vocab.json'))

    def tokenize(self, text):
        tokens = [token.strip() for token in text if token if token.strip()]
        return tokens

    def tokens_to_ids(self, tokens, unk_token='<unk>'):
        unk_idx = self.vocab.get(unk_token, 1)
        ids = [self.vocab.get(token, unk_idx) for token in tokens]
        return ids

    @classmethod
    def build_vocab(cls, texts, output_dir, min_count=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        token_counter = Counter()
        for text in texts:
            text = text.strip().lower()
            tokens = [token for token in text if token.strip()]
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


# experiments utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    def __init__(self, model_dir):
        config_path = os.path.join(model_dir, 'config.json')
        config = load_dict_from_json_file(config_path)
        self.__dict__.update(config)

    def save(self, model_dir):
        config_path = os.path.join(model_dir, 'config.json')
        save_dict_to_json_file(self.__dict__, config_path)

    @property
    def dict(self):
        """Gives dict-like access to Config instance by `config.dict['learning_rate']"""
        return self.__dict__
