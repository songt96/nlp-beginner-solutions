import torch
import logging
from tqdm import trange, tqdm
import numpy as np
import copy
logger = logging.getLogger(__name__)


def train(model, num_train_epochs, train_dataloader, dev_dataloader, train_loss_fn, optimizer,
          metrics_fn, device, patience=1, eval_loss_fn=None):
    logger.info("*** Running training ***")
    logger.info("  Batch size = {}".format(train_dataloader.batch_size))
    if eval_loss_fn is None:
        eval_loss_fn = train_loss_fn
    new_val_loss = 1e7
    best_val_acc = 0
    no_improvment = 0
    best_model = None
    best_val_result = None
    for epoch in trange(int(num_train_epochs)):
        train_loss, train_acc = train_epoch(
            model, train_dataloader, train_loss_fn, optimizer, metrics_fn, device)
        logger.info('Epoch: {}, train loss: {}, acc: {}'.format(
            epoch+1, train_loss, train_acc))
        val_result = evaluate(model, dev_dataloader,
                              eval_loss_fn, metrics_fn, device)
        logger.info('*** Dev result ***')
        for k, v in val_result.items():
            logger.info('{}: {}'.format(k, v))
        val_loss = val_result['loss']
        val_acc = val_result['acc']
        if val_loss < new_val_loss:
            no_improvment = 0
        else:
            no_improvment += 1
            logger.info('No improvment + 1 = {}'.format(no_improvment))
            if no_improvment > patience:
                logger.info('Early stopping in epoch: {}'.format(epoch+1))
                break
        new_val_loss = val_loss

        is_best_acc = val_acc > best_val_acc
        if is_best_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            best_val_result = copy.deepcopy(val_result)
            logger.info('New best acc in validation set in epoch: {}, acc: {}'.format(
                epoch+1, best_val_acc))
    return best_model, best_val_result


def train_epoch(model, dataloader, loss_fn, optimizer, metrics_fn, device):
    train_loss, num_correct, num_examples = 0, 0, 0
    model.train()
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, label = batch[:-1], batch[-1]
        logits = model(*inputs)
        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        train_loss += (loss.item() * batch_size)
        preds = torch.argmax(logits.detach(), dim=1)
        num_correct += torch.sum(preds == label.detach()).item()
        num_examples += batch_size
    train_loss /= num_examples
    train_acc = num_correct / num_examples
    return train_loss, train_acc


def evaluate(model, dataloader, loss_fn, metrics_fn, device):
    logger.info("*** Running evaluation ***")
    logger.info("  Batch size = {}".format(dataloader.batch_size))
    eval_loss, num_examples = 0, 0
    all_pred, all_label = [], []
    model.eval()
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, label = batch[:-1], batch[-1]
        with torch.no_grad():
            logits = model(*inputs)
            loss = loss_fn(logits, label)

        batch_size = label.size(0)
        eval_loss += (loss.item() * batch_size)
        preds = torch.argmax(logits.detach(), dim=1)
        all_pred.append(preds.cpu().numpy())
        all_label.append(label.cpu().numpy())
        num_examples += batch_size
    eval_loss /= num_examples
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    eval_result = metrics_fn(all_pred, all_label)
    eval_result['loss'] = eval_loss
    return eval_result


def predict(model, dataloader, device):
    logger.info("*** Running prediction ***")
    logger.info("Batch size = {}".format(dataloader.batch_size))
    all_pred = []
    model.eval()
    for batch in tqdm(dataloader):
        inputs = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(*inputs)
        preds = torch.argmax(logits.detach(), dim=1)
        all_pred.append(preds.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    return all_pred


def predict_prob(model, dataloader, device):
    logger.info("*** Running prediction ***")
    logger.info("Batch size = {}".format(dataloader.batch_size))
    all_prob = []
    model.eval()
    for batch in tqdm(dataloader):
        inputs = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(*inputs)
        probs = torch.nn.functional.softmax(logits, dim=1)
        all_prob.append(probs.cpu().numpy())
    all_prob = np.concatenate(all_prob, axis=0)
    return all_prob
