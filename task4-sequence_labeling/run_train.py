import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
import models
import trainer
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/msra', type=str,
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='experiments/bilstm', type=str,
                        help="Directory containing params.json")
    parser.add_argument("--output_dir", default=None, type=str,  required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_len", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                        "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--lr", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=15, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--patience', default=1, type=int,
                        help='Patience of early stopping.')
    parser.add_argument('--seed', type=int, default=233,
                        help="random seed for initialization")

    args = parser.parse_args()
    logger.info('Args: {}'.format(args))
    config = utils.Config(args.model_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(args.seed)

    processor = utils.MsraNerProcessor(args.data_dir)
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_test_examples()
    label2idx = processor.get_label2id()

    tokenizer = utils.Tokenizer(args.model_dir)
    train_features = utils.examples_to_features(
        train_examples, tokenizer, label2idx, args.max_len)
    all_label = []
    (all_label.extend(f.token_label_ids) for f in train_features)
    logger.info(np.bincount(all_label))
    dev_features = utils.examples_to_features(
        dev_examples, tokenizer, label2idx, args.max_len)

    train_dataloader = DataLoader(utils.MyDataset(train_features),
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  collate_fn=utils.TokenClfCollate())
    dev_dataloader = DataLoader(utils.MyDataset(dev_features),
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                collate_fn=utils.TokenClfCollate())

    model = models.BiLSTMTagger(config).to(device)
    loss_fn = models.token_clf_ce_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_model, best_val_result = trainer.train(model, args.num_train_epochs,
                                                train_dataloader,
                                                dev_dataloader,
                                                loss_fn, optimizer,
                                                utils.acc_p_r_f1, device,)

    pass
