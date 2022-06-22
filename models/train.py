import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
if '../' not in sys.path:
    sys.path.append('../')
from config import Config


def train(cfg: Config, model: torch.nn.Module, train_loader: DataLoader):
    '''
    params:
        cfg: config
        model: TextCNN
        train_loader: torch.utils.data.Dataloader
    '''
    steps = 0
    for epoch in range(1, cfg.epochs + 1):
        optimizer = cfg.optimizer(model.parameters(), lr=cfg.learning_rate)

        train_bar = tqdm(train_loader)
        for n, data in enumerate(train_bar):
            cat_id, words = data['cat_id'], data['words']

            model.train()
            optimizer.zero_grad()
            logit = model(words)
            loss = cfg.loss_func(logit, cat_id)
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % cfg.log_interval == 0:
                corrects = (torch.max(logit, 1)[
                    1] == torch.max(cat_id.data, 1)[1]).sum()
                # corrects = (torch.max(logit, 1)[1].view(
                #     cat_id.size()).data == cat_id.data).sum()

                accuracy = (100.0 * corrects/cfg.batch_size).item()
                train_bar.set_postfix(
                    {'accuracy': accuracy, 'loss': loss.detach().cpu().numpy()})


def eval(cfg: Config, model: torch.nn.Module, test_loader: DataLoader):
    steps = 0
    for epoch in range(1, cfg.epochs + 1):

        train_bar = tqdm(test_loader)
        for n, data in enumerate(train_bar):
            cat_id, words = data['cat_id'], data['words']

            model.eval()

            logit = model(words)
            loss = cfg.loss_func(logit, cat_id)

            steps += 1

            if steps % cfg.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(
                    cat_id.size()).data == cat_id.data).sum()
                accuracy = 100.0 * corrects/cfg.batch_size
                train_bar.set_postfix(
                    {'accuracy': accuracy, 'loss': loss.detach().cpu().numpy()})
