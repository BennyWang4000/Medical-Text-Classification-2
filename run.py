# %%
from torch.utils.data import DataLoader, random_split
from models.datasets import TextDataset
from models.model import TextCNN
import models.train as train
from math import floor
from config import Config
from torchsummary import summary

cfg = Config()
# if __name__ == '__main__':
# %%
dataset = TextDataset(cfg)

train_num = floor(dataset.__len__() * cfg.train_rate)
test_num = dataset.__len__() - train_num
print('   device:\t', cfg.device)
print('train_num:\t', train_num)
print(' test_num:\t', test_num)
# %%
train_set, val_set = random_split(
    dataset, [train_num, test_num])

train_loader = DataLoader(
    dataset=train_set, batch_size=cfg.batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=val_set, batch_size=cfg.batch_size, shuffle=True)
# # %%
# for i, data in enumerate(train_loader):
#     print(data['cat_id'])
#     print(data['words'].shape)
#     break
# %%
model = TextCNN(cfg)
print(model)
summary(model, (40, 100))
# %%

train.train(cfg=cfg, model=model, train_loader=train_loader)
train.eval(cfg=cfg, model=model, test_loader=test_loader)

# %%
