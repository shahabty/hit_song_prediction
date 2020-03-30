import torch
import torch.nn as nn
import math


def basic_block(D_in,D_out):
    return torch.nn.Sequential(
        nn.Linear(D_in,D_out),
        nn.ReLU(),
        nn.BatchNorm1d(D_out),
        )

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class high_level(nn.Module):
    def __init__(self):
        super(high_level,self).__init__()
        D_in,D_out = [1603,1024,256,32,16,4,1],[1024,256,32,16,4,1]
        layer = []

        for i in range(len(D_out)):
            layer.append(basic_block(D_in[i],D_out[i]))

        self.encoder = nn.Sequential(*layer)

        self.encoder.apply(init_weights)

    def forward(self,input_data):
        return self.encoder(input_data)



class low_level(nn.Module):
    def __init__(self):
        super(low_level,self).__init__()
    
    def forward(self,input_data):
        return input_data



class hybrid(nn.Module):
    def __init__(self):
        super(hybrid,self).__init__()
        self.high_level_net = high_level()
#        low_level_net = low_level()

    def forward(self,input_data):
        return self.high_level_net(input_data)
