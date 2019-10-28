import logging
import copy
import numpy as np
from tqdm import tqdm, trange
import torch
logger = logging.getLogger(__name__)


def train(model, num_train_epochs, train_dataloader, dev_dataloader,
          loss_fn, optimizer, metrics_fn, device, patience=1):
    logger.info("*** Running training ***")
    logger.info("Batch size: {}".format(train_dataloader.batch_size))
    new_val_loss = 1e7
    best_val_acc, no_improvment = 0, 0
    best_model, best_val_result = None, None
    for epoch in trange(num_train_epochs):
        train_loss, train_acc = train_epoch(
            model, train_dataloader, loss_fn, optimizer, device)
        logger.info('Epoch: {}, train loss: {}, acc: {}'.format(
            epoch + 1, train_loss, train_acc))
        val_result = evaluate(model, dev_dataloader,
                              loss_fn, metrics_fn, device)
        logger.info('*** Dev result ***')
        for k, v in val_result.items():
            logger.info('{:<4}: {}'.format(k, v))
        val_loss = val_result['loss']
        val_acc = val_result['acc']
        if val_loss < new_val_loss:
            no_improvment = 0
        else:
            no_improvment += 1
            logger.info('No improvment + 1 = {}'.format(no_improvment))
            if no_improvment > patience:
                logger.info('Early stopping in epoch: {}'.format(epoch + 1))
                break
        new_val_loss = val_loss

        is_best_acc = val_acc > best_val_acc
        if is_best_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            best_val_result = copy.deepcopy(val_result)
            logger.info(
                'New best acc in validation set in epoch: {}, acc: {}'.format(
                    epoch + 1, best_val_acc))
    return best_model, best_val_result


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    train_loss, num_correct, num_tokens = 0, 0, 0
    model.train()
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch[:-1], batch[-1]
        logits = model(*inputs)
        mask = labels.view(-1) > 0
        loss = loss_fn(logits, labels, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_batch_tokens = torch.sum(mask).item()
        train_loss += (loss.item() * num_batch_tokens)
        preds = torch.argmax(logits.detach(), dim=-1)
        num_correct += torch.sum((preds == labels.detach()
                                  ).view(-1)[mask]).item()
        num_tokens += num_batch_tokens
    train_loss /= num_tokens
    train_acc = num_correct / num_tokens
    return train_loss, train_acc


def evaluate(model, dataloader, loss_fn, metrics_fn, device):
    logger.info("*** Running evaluation ***")
    logger.info("Batch size: {}".format(dataloader.batch_size))
    all_pred, all_label = [], []
    val_loss, num_tokens = 0, 0
    model.eval()
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch[:-1], batch[-1]
        with torch.no_grad():
            logits = model(*inputs)
            mask = labels.view(-1) > 0
            loss = loss_fn(logits, labels, mask)

        num_batch_tokens = torch.sum(mask).item()
        val_loss += (loss.item() * num_batch_tokens)
        preds = torch.argmax(logits.detach(), dim=-1)
        all_pred.append(preds.view(-1).cpu().numpy())
        all_label.append(labels.view(-1).cpu().numpy())
        num_tokens += num_batch_tokens
    all_pred = np.concatenate(all_pred, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_mask = all_label.reshape(-1) > 0
    val_loss /= num_tokens
    val_result = metrics_fn(all_pred, all_label, all_mask)
    val_result['loss'] = val_loss
    return val_result


def predict(model, dataloader, device):
    logger.info("*** Running prediction ***")
    logger.info("Batch size: {}".format(dataloader.batch_size))
    all_pred = []
    model.eval()
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = batch
        with torch.no_grad():
            logits = model(*inputs)

        preds = torch.argmax(logits.detach(), dim=-1)
        all_pred.append(preds.view(-1).cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    return all_pred
