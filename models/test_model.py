from torch import nn
from .attention import Seq_Transformer
# This file aims to embed the cycle's data.


class Encode_Model(nn.Module):
    def __init__(self):
        super(Encode_Model, self).__init__()
        self.num_channels = 17
        # self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4,
        #                                        heads=4, mlp_dim=64)
        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=100, depth=4,
                                               heads=4, mlp_dim=64)

    def forward(self, x_in):
        context = self.seq_transformer(x_in)
        return context

