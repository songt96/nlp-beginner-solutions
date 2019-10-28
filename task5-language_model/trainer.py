import copy
import argparse
import logging
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import models
logger = logging.getLogger(__name__)


def train(model, num_epochs, train_dataloader, dev_dataloader,
          loss_fn, optimizer, metrics_fn, device):

    pass


def train_epoch(model, dataloader, loss_fn, optimzier, device):
    pass


def evaluate(model, dataloader, loss_fn, metrics, device):
    pass


def predict(model, dataloader, device):
    pass
