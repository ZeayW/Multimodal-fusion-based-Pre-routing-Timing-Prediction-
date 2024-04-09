import torch as th
from torch import nn
from dgl import function as fn


def cell_msg_reduce(nodes):
    return {'h_neigh1': th.max(nodes.mailbox['m'], dim=1)}


class MLP(th.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(th.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(th.nn.LeakyReLU(negative_slope=0.0))
                # fcs.append(torch.nn.ReLU())
                if dropout: fcs.append(th.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(th.nn.BatchNorm1d(sizes[i]))
        self.layers = th.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class PathConv(nn.Module):

    def __init__(self,
                 out_feat_dim,
                 hidden_feat_dim,
                 cell_feat_dim,
                 net_feat_dim,
                 flag_attn=False,
                 num_heads=1,
                 activation=th.nn.functional.relu,
                 # activation = th.nn.LeakyReLU(),
                 bias=True,
                 norm=None):
        super(PathConv, self).__init__()
        self.flag_attn = flag_attn
        self.hidden_feat_dim = hidden_feat_dim
        self.out_feat_dim = out_feat_dim
        self.cell_feat_dim = cell_feat_dim
        self.net_feat_dim = net_feat_dim
        self.num_heads = num_heads

        # self.mlp_cell_feat = MLP(cell_feat_dim, hidden_feat_dim , hidden_feat_dim, hidden_feat_dim)
        # self.mlp_net_feat = MLP(net_feat_dim, hidden_feat_dim,hidden_feat_dim, hidden_feat_dim)
        # self.mlp_cell_combine = MLP(out_feat_dim + hidden_feat_dim, (out_feat_dim+hidden_feat_dim), self.out_feat_dim)
        # self.mlp_net_combine = MLP(out_feat_dim + hidden_feat_dim, (out_feat_dim + hidden_feat_dim),self.out_feat_dim)
        # self.mlp_pi = MLP(cell_feat_dim, cell_feat_dim, self.out_feat_dim)

        self.fc_cell_neigh = MLP(self.hidden_feat_dim, 256,self.out_feat_dim)
        self.fc_cell_self = MLP(self.cell_feat_dim, 256,self.out_feat_dim)
        self.fc_net_self = MLP(self.net_feat_dim, 256,self.out_feat_dim)

        # net_indim = self.net_feat_dim + self.out_feat_dim
        # self.mlp_net = MLP(net_indim, net_indim * 2, self.out_feat_dim)
        # cell_indim = self.cell_feat_dim + self.out_feat_dim
        # self.mlp_cell = MLP(cell_indim, cell_indim * 2, self.out_feat_dim)
        # self.mlp_pi = MLP(self.cell_feat_dim, self.cell_feat_dim * 2, self.out_feat_dim)
        # self.fc_cell_neigh = MLP(self.in_feat_dim, self.out_feat_dim, bias=bias)
        # self.fc_cell_self = MLP(self.cell_feat_dim, self.out_feat_dim, bias=bias)
        # self.fc_net_self = MLP(self.net_feat_dim, self.out_feat_dim, bias=bias)
        # self.fc_cell_neigh = nn.Linear(self.in_feat_dim, self.out_feat_dim, bias=bias)
        # self.fc_cell_self = nn.Linear(self.cell_feat_dim, self.out_feat_dim, bias=bias)
        # self.fc_net_neigh = nn.Linear(self.in_feat_dim, self.out_feat_dim, bias=bias)
        # self.fc_net_self = nn.Linear(self.net_feat_dim, self.out_feat_dim, bias=bias)

        negative_slope = 0.2
        if self.flag_attn:
            self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, self.out_feat_dim)))
        # self.leaky_relu = nn.LeakyReLU(negative_slope)

        # set some attributes
        self.activation = activation
        self.norm = norm
        # initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        """
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.fc_net_neigh.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_net_self.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_cell_neigh.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_cell_self.weight, gain=gain)
        if self.flag_attn:
            nn.init.xavier_normal_(self.attn, gain=gain)

    def apply_net_func(self, nodes):

        """

        An apply function to further update the node features of output pins
        after the message reduction.

        :param nodes: the applied nodes
        :return:
            {'msg_name':msg}
                msg: torch.Tensor
                    The aggregated messages of shape :math:`(N, D_{out})` where : N is number of nodes, math:`D_{out}`
                    is size of messages.
        """
        #
        h_self = self.fc_net_self(nodes.data['net_feat'])
        h_drive = nodes.data['h_neigh1']
        h = h_self + h_drive
        #
        # h_self = self.mlp_net_feat(nodes.data['net_feat'])
        # h = th.cat([h_self, nodes.data['h_neigh1']], dim=1)
        # h = self.mlp_net_combine(h)


        return {'h': h}

    def cell_msg_reduce(self, nodes):
        # print(th.max(nodes.mailbox['m'], dim=1).values)
        # exit()
        msg = nodes.mailbox['m']
        weight = th.softmax(msg, dim=1)
        return {'h_neigh1': (msg * weight).sum(1)}

    def apply_cell_func(self, nodes):

        """

        An apply function to further update the node features of output pins
        after the message reduction.

        :param nodes: the applied nodes
        :return:
            {'msg_name':msg}
               msg: torch.Tensor
                   The aggregated messages of shape :math:`(N, D_{out})` where : N is number of nodes, math:`D_{out}`
                   is size of messages.
        """

        h_self = self.fc_cell_self(nodes.data['cell_feat'])
        h_input = self.fc_cell_neigh(nodes.data['h_neigh1'])
        h = h_self + h_input

        # h_self = self.mlp_cell_feat(nodes.data['cell_feat'])
        # h = th.cat([h_self, nodes.data['h_neigh1']], dim=1)
        # h = self.mlp_cell_combine(h)

        return {'h': h}

    def apply_cell_func_level0(self, nodes):
        h = self.fc_cell_self(nodes.data['cell_feat'])

        # h = self.mlp_pi(nodes.data['cell_feat'])
        return {'h': h}

    def copy_src(self, edges):
        return {'m': edges.src['h']}

    def cal_edge_attn(self, edges):
        e = self.activation(
            (edges.src['h'].view((-1, 1, self.out_feat_dim)) * self.attn).sum(dim=-1)
        )
        return {'e': e}

    def forward(self, graph, cur_nodes, eids, targets, level_id):
        r"""

        Description
        -----------
        Compute TimeGNN layer.

        Parameters
        ----------
        graph : heterograph
            The graph, with two edge type: cell/net
        level_id: int
            The index of current topological level
        cur_nodes: List[int]
            list of node id in current topological level

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        # the nodes in the same level are of the same type: either cell or net
        # and the two type changes alternatively
        #   e.g., cell, net, cell, net, ...
        # so that when the level id is even, the type is cell, else net

        if level_id % 2 == 1:
            graph.pull(cur_nodes, fn.copy_src('h', 'm'), fn.mean('m', 'h_neigh1'),
                       apply_node_func=self.apply_net_func, etype='net')

        else:
            if self.flag_attn and level_id != 0:
                # calculate edge attention
                graph.apply_edges(func=self.cal_edge_attn, edges=eids, etype='cell')
                # apply edge softmax
                # our version is based on sparse matrix softmax
                e = graph.edges['cell'].data['e'][eids]
                target_edges = (graph.edges(etype='cell')[0][eids], graph.edges(etype='cell')[1][eids])
                i = th.stack(target_edges, 0)
                sp = th.sparse.FloatTensor(i, e)
                a = th.sparse.softmax(sp, 0).coalesce().values().view((-1, 1))
                graph.edges['cell'].data['a'][eids] = a
                # message passing
                graph.pull(cur_nodes, fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_neigh1'),
                           apply_node_func=self.apply_cell_func, etype='cell')
            else:
                apply_node_func = self.apply_cell_func_level0 if level_id == 0 else self.apply_cell_func
                #apply_node_func = self.apply_cell_func
                reduce_func = fn.max('m', 'h_neigh1') if level_id == 0 else self.cell_msg_reduce
                graph.pull(cur_nodes, fn.copy_src('h', 'm'), reduce_func=reduce_func,
                           apply_node_func=apply_node_func, etype='cell')

                # change the apply_func for PIs
                # graph.pull(cur_nodes, fn.copy_src('h','m'), fn.max('m','h_neigh1'),
                #             apply_node_func=apply_node_func, etype='cell')

                # orignal one
                # graph.pull(cur_nodes, fn.copy_src('h','m'), fn.max('m','h_neigh1'),
                #             apply_node_func=self.apply_cell_func, etype='cell')

        # activation
        # print(graph.nodes['pin'].data['h'][cur_nodes])
        if self.activation is not None:
            graph.nodes['pin'].data['h'][cur_nodes] = self.activation(graph.nodes['pin'].data['h'][cur_nodes])
        # normalization
        if self.norm is not None:
            graph.nodes['pin'].data['h'][cur_nodes] = self.norm(graph.nodes['pin'].data['h'][cur_nodes])
        #
        # if level_id==0:
        #     print(graph.nodes['pin'].data['h'][cur_nodes])
        # print(graph.nodes['pin'].data['h'][cur_nodes])
        return graph.nodes['pin'].data['h'][targets]


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
        self.mlp_alpha = MLP(1,128,64)

    def forward(self, graph,nodes,eids,target_list,level_id,level_id_th,path_map):
        factor = 1
        # factor = th.sigmoid(level_id_th)
        h_cnn = factor*self.fcn[0](path_map) \
                  if self.fcn[0] is not None and len(target_list)!=0 \
                  else None

        h_gnn = self.gnn[0](graph,nodes,eids,target_list,level_id) \
                   if self.gnn[0] is not None \
                   else None

        h_global = self.mlp_alpha(level_id_th).expand(len(target_list),64)
        if len(target_list)==0:
            return None

        if h_cnn is None:
            h =h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn,h_cnn,h_global),1)

        return self.mlp[0](h).squeeze(-1)

        return h

