# %%
from torch.utils.data import DataLoader, random_split
from models.datasets import TextDataset
from models.model import TextCNN
import models.train as train
from math import floor
from torchsummary import summary

from config import *
# %%
if __name__ == '__main__':
    dataset = TextDataset(data_path=DATA_PATH,
                          w2v_model_path=W2V_MODEL_PATH, weight_path=WEIGHT_PATH, words_len=WORDS_LEN, embed_dim=EMBED_DIM, device=DEVICE)

    train_num = floor(dataset.__len__() * TRAIN_RATE)
    test_num = dataset.__len__() - train_num
    print('   device:\t', DEVICE)
    print('train_num:\t', train_num)
    print(' test_num:\t', test_num)

    train_set, val_set = random_split(
        dataset, [train_num, test_num])

    train_loader = DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    model = TextCNN(embed_dim=EMBED_DIM, class_num=CLASS_NUM,
                    kernel_num=KERNEL_NUM, kernel_sizes=KERNEL_SIZES, dropout=DROPOUT).to(DEVICE)

    if IS_WANDB:
        import wandb
        wandb.init(project="MedicalClassification", entity="bennywang4000")
        wandb.watch(model)

    print(model)
    summary(model, (40, 100))

    for epoch in range(1, EPOCHS + 1):
        print('epoch:\t', epoch)
        train.train(model=model, train_loader=train_loader, epochs=EPOCHS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                    loss_func=LOSS_FUNC, log_interval=LOG_INTERVAL, batch_size=BATCH_SIZE, saving_path=SAVING_PATH, is_wandb=IS_WANDB)
        train.eval(model=model, test_loader=test_loader, epochs=EPOCHS,
                   loss_func=LOSS_FUNC, log_interval=LOG_INTERVAL, batch_size=BATCH_SIZE, is_wandb=IS_WANDB)
