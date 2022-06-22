from torch import device, cuda, optim
import torch.nn.functional as F


class Config:
    def __init__(self):
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
        self.data_path = 'D:\CodeRepositories\py_project\data_mining\data\df_jieba_none_remove.csv'
        self.weight_path = 'D:\CodeRepositories\py_project\data_mining\data\info_gain_none_weight.csv'
        self.w2v_model_path = 'D:\CodeRepositories\py_project\data_mining\data\word2vec_jieba_none.model'
        self.saving_path = 'D:\CodeRepositories\py_project\data_mining\data\runs'

        self.train_rate = 0.8
        self.is_valid = True
        self.epochs = 5
        self.batch_size = 4
        self.words_len = 40
        self.learning_rate = 0.0001
        self.optimizer = optim.Adam
        self.loss_func = F.cross_entropy
        self.log_interval = 10

        self.embed_dim = 100
        self.class_num = 2
        self.kernel_num = 100
        self.kernel_sizes = [3, 4, 5]
        self.dropout = 0.5

        self.is_cuda = cuda.is_available()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
