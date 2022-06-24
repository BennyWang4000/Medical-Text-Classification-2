from torch import device, cuda, optim
import torch.nn.functional as F


# class Config:
#     def __init__(self):
'''config class
data config:
    data_path
    weight_path
    w2v_model_path
    saving_path

train config:
    train_per
    is_valid
    epochs
    words_len
    learning_rate
    optimizer
    loss_func
    log_interval

model config:
    embed_dim
    class_num
    kernel_num
    kernel_sizes
    dropout

torch_init:
    is_cuda
    device
'''

DATA_PATH = '../input/medical-question-corpus/medical_question_corpus/df_jieba_none_remove.csv'
WEIGHT_PATH = '../input/medical-question-corpus/medical_question_corpus/info_gain_none_weight.csv'
W2V_MODEL_PATH = '../input/medical-question-corpus/medical_question_corpus/word2vec_jieba_none.model'
SAVING_PATH = './'

TRAIN_RATE = 0.5
IS_VALID = True
EPOCHS = 1
BATCH_SIZE = 8
WORDS_LEN = 40
LEARNING_RATE = 0.0002
OPTIMIZER = optim.Adam
LOSS_FUNC = F.cross_entropy
LOG_INTERVAL = 10

EMBED_DIM = 100
CLASS_NUM = 2
KERNEL_NUM = 200
KERNEL_SIZES = [2, 3, 4, 5]
DROPOUT = 0.5


IS_CUDA = cuda.is_available()
DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
