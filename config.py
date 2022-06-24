from torch import device, cuda, optim
import torch.nn.functional as F

DATA_PATH = '../data/df_jieba_none_remove.csv'
WEIGHT_PATH = '../data/info_gain_none_weight.csv'
W2V_MODEL_PATH = '../data/word2vec_jieba_none.model'
SAVING_PATH = '../runs'

TRAIN_RATE = 0.5
IS_VALID = True
EPOCHS = 1
BATCH_SIZE = 8
WORDS_LEN = 40
LEARNING_RATE = 0.0002
OPTIMIZER = optim.Adam
LOSS_FUNC = F.cross_entropy
LOG_INTERVAL = 10

IS_WANDB= False

EMBED_DIM = 100
CLASS_NUM = 2
KERNEL_NUM = 200
KERNEL_SIZES = [2, 3, 4, 5]
DROPOUT = 0.5

IS_CUDA = cuda.is_available()
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")