import os
import argparse
import logging
import utils
import models
import trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/lcqmc', type=str,
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='./experiments/esim_char', type=str,
                        help="Directory containing params.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--max_len", default=48, type=int,
                        help="The maximum total input sequence length after tokenization. \n"
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

    processor = utils.LcqmcProcessor(args.data_dir, )
    label2id = processor.get_label2id()

    tokenizer = utils.CharTokenizer(args.model_dir)
    config.vocab_size = len(tokenizer.vocab)
    embedding = utils.get_embedding(
        tokenizer.vocab, config.embedding_path, config.embedding_dim)
    model = models.Esim(
        config, torch.FloatTensor(embedding)).to(device)
    logger.info(model)
    loss_fn = nn.CrossEntropyLoss()
    if args.do_train:
        train_examples = processor.get_train_examples()
        dev_examples = processor.get_dev_examples()
        train_features = utils.examples_to_features(
            train_examples, label2id, tokenizer,
            max_len=args.max_len, verbose=True)
        dev_features = utils.examples_to_features(
            dev_examples, label2id, tokenizer,
            max_len=args.max_len, verbose=True)
        train_dataloader = DataLoader(utils.MyDataset(train_features),
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      collate_fn=utils.TextPairCollate())
        dev_dataloader = DataLoader(utils.MyDataset(dev_features),
                                    batch_size=args.eval_batch_size,
                                    shuffle=False,
                                    collate_fn=utils.TextPairCollate())
        optimizer = optim.Adam(model.parameters(), lr=args.lr,)
        best_model, best_val_result = trainer.train(
            model, args.num_train_epochs, train_dataloader, dev_dataloader,
            loss_fn, optimizer, utils.acc_p_r_f1, device)
        utils.save_experiment(config, best_model,
                              best_val_result, args.output_dir)
        tokenizer.save_vocab(args.output_dir)
    if args.do_eval:
        if args.do_train:
            args.model_dir = args.output_dir
        test_examples = processor.get_test_examples()
        test_features = utils.lcqmc_examples_to_features(
            test_examples, label2id, tokenizer,
            max_len=args.max_len, verbose=True)
        test_dataloader = DataLoader(utils.MyDataset(test_features),
                                     batch_size=args.eval_batch_size,
                                     shuffle=False,
                                     collate_fn=utils.TextPairCollate())
        model.load_state_dict(torch.load(
            os.path.join(args.model_dir, 'pytorch_model.bin')))
        test_result = trainer.evaluate(model, test_dataloader,
                                       loss_fn, utils.acc_p_r_f1,
                                       device, )
        logger.info('*** Test result ***')
        for k, v in test_result.items():
            logger.info('{}: {}'.format(k, v))
