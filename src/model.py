import torch as th
import torch.nn as nn
import torch.nn.functional as F



from time import time


class MLP(nn.Module):
    r"""
                Description
                -----------
                a simple multilayer perceptron
    """
    def __init__(self,in_dim,out_dim,nlayers,activation =nn.ReLU() ,dropout=0.5):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers= nn.Sequential()
        dim1 = in_dim
        for i in range(nlayers-1):
            self.layers.add_module('dropout_{}'.format(i+1),self.dropout)
            self.layers.add_module('activation_{}'.format(i+1), self.activation)
            self.layers.add_module('linear_{}'.format(i+1),nn.Linear(dim1, int(dim1/2)))
            dim1 = int(dim1 / 2)
        self.layers.add_module('linear_{}'.format(nlayers),nn.Linear(dim1, out_dim))
    def forward(self,embedding):
        return self.layers(embedding).squeeze(-1)

class MLP2(nn.Module):
    r"""
                Description
                -----------
                a simple multilayer perceptron
    """
    def __init__(self,in_dim,out_dim,nlayers,activation =nn.ReLU() ,dropout=0.5):
        super(MLP2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers= nn.Sequential(
            nn.Linear(in_dim,2*in_dim),
            self.activation,
            nn.Linear(2*in_dim,out_dim),
        )
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.layers[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.layers[2].weight, gain=gain)

    def forward(self,embedding):
        return self.layers(embedding).squeeze(-1)

class LayoutNet3(nn.Module):
    def __init__(self,pooling):
        super(LayoutNet3, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
            pooling_layer2 = nn.MaxPool2d(2, 2, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2,2,0)
            pooling_layer2 = nn.AvgPool2d(2,2,0)
        else:
            assert False, 'wrong pooling type for layoutnet!'

        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(64, 128, 9, 1, 4),
            nn.ReLU(),
            #nn.Conv2d(32, 1, 7, 1, 3),
            )
        
        self.decode = nn.Sequential(
            nn.Conv2d(128, 256, 7, 1, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 9, 2, 4, 1),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
            pooling_layer2,
            nn.ReLU()
        )

 
    def forward(self, x):
        encode_out = self.encode(x)      
        out = self.decode(encode_out)
        return out

class LayoutNet2(nn.Module):
    def __init__(self,pooling):
        super(LayoutNet2, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
            pooling_layer2 = nn.MaxPool2d(4, 4, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2,2,0)
            pooling_layer2 = nn.AvgPool2d(4,4,0)
        else:
            assert False, 'wrong pooling type for layoutnet!'

        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(64, 32, 9, 1, 4),
            nn.ReLU(),
            #nn.Conv2d(32, 1, 7, 1, 3),
            )
        
        self.decode = nn.Sequential(
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 9, 2, 4, 1),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, 5, 2, 2, 1),
            nn.Conv2d(4, 1, 3, 1, 1),
            pooling_layer2,
            nn.ReLU()
        )
        

 
    def forward(self, x):
        encode_out = self.encode(x)      
        out = self.decode(encode_out)
        return out
        
class LayoutNet(nn.Module):
    def __init__(self,pooling):
        super(LayoutNet, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2,2,0)
        else:
            assert False, 'wrong pooling type for layoutnet!'
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            #nn.ReLU(),
            pooling_layer,
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, 1, 3),
            #nn.ReLU(),
            pooling_layer,
            nn.ReLU(),
            nn.Conv2d(64, 32, 9, 1, 4),
            nn.ReLU(),
            nn.Conv2d(32, 1, 7, 1, 3),
            nn.ReLU())

        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.encode[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.encode[3].weight, gain=gain)
        # nn.init.xavier_uniform_(self.encode[6].weight, gain=gain)
        # nn.init.xavier_uniform_(self.encode[8].weight, gain=gain)

    def forward(self, x):
        out = self.encode(x)
        return out

class PathModel(nn.Module):
    r"""
                    Description
                    -----------
                    the model used to classify
                    consits of two GNN models and one MLP model
        """
    def __init__(
        self, gnn,fcn,mlp
    ):
        super(PathModel, self).__init__()
        
        self.gnn = nn.Sequential(gnn)
        self.fcn = nn.Sequential(fcn)
        self.mlp = nn.Sequential(mlp)

    def forward(self, graph,nodes,eids,target_list,level_id,path_map):
        
        h_cnn = self.fcn[0](path_map) \
                  if self.fcn[0] is not None and len(target_list)!=0 \
                  else None
        h_gnn = self.gnn[0](graph,nodes,eids,target_list,level_id) \
                   if self.gnn[0] is not None \
                   else None

        if len(target_list)==0:
            return None

        if h_cnn is None:
            h =h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn,h_cnn),1)

        return self.mlp[0](h)

        if self.gnn[0] is None:
            path_map = path_map.view(-1,128*128)
            h = self.fcn[0](path_map)
        elif self.fcn[0] is None:
            h = self.gnn[0](graph,nodes,target_list,level_id)
        else:
            h_gnn = self.gnn[0](graph,nodes,target_list,level_id)
            h_cnn = self.fcn[0](path_map)
            
            # combine the information from both direction
            h = th.cat((h_gnn,h_cnn),1)

        h = self.mlp[0](h)

        return h

