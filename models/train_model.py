import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import nn


import traceback
import copy
import datetime

from sklearn.metrics import accuracy_score

from torch.nn import functional as F


class TrainModel(object):
    def __init__(self, config):
        """Train a model."""

        self._lr = config.get('lr', 1.0e-3)
        self._step_size_scheduler = config.get('step_size_scheduler', 10)
        self._gamma_scheduler = config.get('gamma_scheduler', 0.1)
        self._early_stopping_patience = config.get(
            'early_stopping_patience', 10)
        self._batch_size = config.get('batch_size', 32)
        self._epoch_n = config.get('epoch_n', 100)
        self._criterion = config.get('criterion', F.cross_entropy)

    def train(self, model, train_dataset, val_dataset):

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        lr_scheduler = StepLR(
            optimizer, step_size=self._step_size_scheduler, gamma=self._gamma_scheduler)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self._batch_size, shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        best_val_loss = float('inf')
        best_epoch_i = 0
        best_model = copy.deepcopy(model)

        for epoch_i in range(self._epoch_n):
            try:
                epoch_start = datetime.datetime.now()
                print('Epoch {}/{}'.format(epoch_i, self._epoch_n))
                print('lr_value: {}'.format(
                    lr_scheduler.get_lr()[0]), flush=True)
                lr_scheduler.step()
                model.train()
                mean_train_loss = 0
                mean_train_acc = 0
                train_batches_n = 0

                for batch_x, batch_y in train_dataloader:

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()

                    pred = model(batch_x)

                    loss = self._criterion(pred, batch_y)
                    acc = accuracy_score(pred.detach().cpu().numpy().argmax(-1),
                                         batch_y.cpu().numpy())
                    loss.backward()
                    optimizer.step()

                    mean_train_loss += float(loss)
                    mean_train_acc += float(acc)
                    train_batches_n += 1

                mean_train_loss /= train_batches_n
                mean_train_acc /= train_batches_n
                print('{:0.2f} s'.format(
                    (datetime.datetime.now() - epoch_start).total_seconds()))
                print('Train Loss', mean_train_loss)
                print('Train Acc', mean_train_acc)

                model.eval()
                mean_val_loss = 0
                mean_val_acc = 0
                val_batches_n = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_dataloader:

                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)

                        pred = model(batch_x)
                        loss = self._criterion(pred, batch_y)
                        acc = accuracy_score(pred.detach().cpu().numpy().argmax(-1),
                                             batch_y.cpu().numpy())

                        mean_val_loss += float(loss)
                        mean_val_acc += float(acc)
                        val_batches_n += 1

                mean_val_loss /= val_batches_n
                mean_val_acc /= val_batches_n
                print('Val Loss', mean_val_loss)
                print('Val Acc', mean_val_acc)

                if mean_val_loss < best_val_loss:
                    best_epoch_i = epoch_i
                    best_val_loss = mean_val_loss
                    best_model = copy.deepcopy(model)
                    print('New best model!')
                elif epoch_i - best_epoch_i > self._early_stopping_patience:
                    print('The model has not improved over the last {} epochs, stop learning.'.format(
                        self._early_stopping_patience))
                    break
                print()
            except KeyboardInterrupt:
                print('Stopped by user.')
                break
            except Exception as ex:
                print('Error: {}\n{}'.format(ex, traceback.format_exc()))
                break

        return best_val_loss, best_model


if __name__ == '__main__':
    train()
