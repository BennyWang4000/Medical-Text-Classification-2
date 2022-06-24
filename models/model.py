import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, **cfg):
        '''
        params:
            embed_dim
            class_num
            kernel_num
            kernel_sizes
            dropout
        '''
        super(TextCNN, self).__init__()
        self.cfg = cfg

        # V = cfg['embed_num']
        D = cfg['embed_dim']
        C = cfg['class_num']
        Ci = 1
        Co = cfg['kernel_num']
        Ks = cfg['kernel_sizes']

        # self.w2v_model = word2vec.Word2Vec.load(w2v_model_path)

        # self.embed = nn.Embedding(V, D)
        # self.embed = nn.Embedding.from_pretrained(
        #     torch.FloatTensor(self.w2v_model.wv.vectors))

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(cfg['dropout'])

        self.fc2 = nn.Linear(len(Ks) * Co, len(Ks) * int(Co / 4))
        self.fc1 = nn.Linear(len(Ks) * int(Co / 4), C)

        # if self.cfg['static']:
        #     self.embed.weight.requires_grad = False

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc2(x)  # (len(Ks)*Co, len(Ks)*Co/4)
        logit = self.fc1(x)  # (N, C)
        return logit
