import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import os


def train(model: torch.nn.Module, train_loader: DataLoader, is_wandb, **cfg):
    '''
    params:
        model: TextCNN
        train_loader: torch.utils.data.Dataloader
        epochs
        optimizer
        learning_rate
        loss_func
        log_intreval
        batch_size
        saving_path
    '''
    steps = 0
    all_corrects = 0
    all_batch = 0
    # for epoch in range(1, cfg['epochs'] + 1):
    optimizer = cfg['optimizer'](
        model.parameters(), lr=cfg['learning_rate'])

    train_bar = tqdm(train_loader)
    for n, data in enumerate(train_bar):
        cat_id, words = data['cat_id'], data['words']
        batch_size = words.shape[0]
        model.train()
        optimizer.zero_grad()
        logit = model(words)
        loss = cfg['loss_func'](logit, cat_id)
        loss.backward()
        optimizer.step()

        if is_wandb:
            wandb.log({"train_loss": loss})

        steps += 1

        if steps % cfg['log_interval'] == 0 or steps == len(train_loader):
            corrects = (torch.max(logit, 1)[
                1] == torch.max(cat_id.data, 1)[1]).sum().item()
            all_corrects += corrects
            all_batch += batch_size
            accuracy = 100.0 * corrects / batch_size
            all_accuracy = 100.0 * all_corrects / all_batch

            if is_wandb:
                wandb.log({"train_accuracy": all_accuracy})

            train_bar.set_postfix(
                {'accuracy': accuracy, 'a_accuracy': all_accuracy, 'loss': loss.detach().cpu().numpy()})

        if steps % 5000 == 0 or steps == len(train_loader):
            torch.save(model.state_dict(), os.path.join(
                cfg.saving_path, 'model_' + str(steps) + '.pt'))


def eval(model: torch.nn.Module, test_loader: DataLoader, **cfg):
    '''
    params:
        epochs
        loss_func
        log_interval
        batch_size
    '''
    steps = 0
    all_corrects = 0
    all_batch = 0

    train_bar = tqdm(test_loader)
    for n, data in enumerate(train_bar):
        cat_id, words = data['cat_id'], data['words']
        batch_size = words.shape[0]

        model.eval()

        logit = model(words)
        loss = cfg['loss_func'](logit, cat_id)

        wandb.log({"test_loss": loss})

        steps += 1

        if steps % cfg['log_interval'] == 0 or steps == len(test_loader):
            corrects = (torch.max(logit, 1)[1].view(
                cat_id.size()).data == cat_id.data).sum()
            all_corrects += corrects
            all_batch += batch_size
            accuracy = 100.0 * corrects / batch_size
            all_accuracy = 100.0 * all_corrects / all_batch

            wandb.log({"test_accuracy": all_accuracy})

            train_bar.set_postfix(
                {'accuracy': accuracy, 'a_accuracy': all_accuracy, 'loss': loss.detach().cpu().numpy()})
