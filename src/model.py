import torch as th
from torch import nn
from dgl import function as fn
import torch.nn.functional as F

def cell_msg_reduce(nodes):
    return {'h_neigh1': th.max(nodes.mailbox['m'], dim=1)}


class MLP(th.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False,negative_slope=0):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(th.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(th.nn.LeakyReLU(negative_slope=negative_slope))
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

        self.fc_cell_neigh = MLP(self.hidden_feat_dim, 256,self.out_feat_dim)

        self.fc_cell_self = MLP(self.cell_feat_dim, 256,self.out_feat_dim)
        self.fc_net_self = MLP(self.net_feat_dim, 256,self.out_feat_dim)
        self.fc_net_drive = MLP(2,self.out_feat_dim)

        self.fc_attn2 = nn.Linear(self.out_feat_dim, 1, bias=False)

        if flag_attn:
            dim_key = 256
            self.fc_key = nn.Linear(1, dim_key, bias=False)
            self.fc_attn = nn.Linear(2*dim_key, 1, bias=False)

        # set some attributes
        self.activation = activation
        self.norm = norm


    def apply_netdrive_func(self, nodes):

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
        # h= self.fc_net_drive(nodes.data['h_drive'])
        h = nodes.data['h_drive']
        return {'h_drive': h}

    def message_func_net(self, edges):
        return {'m': edges.src['h'], 'm_d': edges.src['h_drive']}

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
        m_self = nodes.data['net_feat']
        h_self = self.fc_net_self(m_self)
        # h_self = self.fc_net_self(nodes.data['net_feat'])
        h_drive = nodes.data['h_neigh1']

        h = h_self + h_drive


        return {'h': h}

    def cell_msg_reduce(self, nodes):
        msg = nodes.mailbox['m']
        weight = th.softmax(msg, dim=1)
        return {'h_neigh1': (msg * weight).sum(1)}

    def cell_msg_reduce_attn2(self, nodes):
        msg = nodes.mailbox['m']
        key = self.fc_attn2(msg)
        weight = th.softmax(key, dim=0)
        # print(key,weight)
        return {'h_neigh1': (msg * weight).sum(1)}

    def cell_msg_reduce_attn(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = th.sum(alpha * nodes.mailbox['m'], dim=1)

        return {'h_neigh1': h}


    def message_func_attn(self,edges):
        z = th.cat([self.fc_key(edges.src['key']), self.fc_key(edges.dst['key'])], dim=1)
        #z = self.fc_key(edges.src['key'])
        a = self.fc_attn(z)
        return {'m':edges.src['h'],'e':F.leaky_relu(a)}

    def apply_cell_func(self, nodes):

        cell_feat = nodes.data['cell_feat']
        h_self = self.fc_cell_self(cell_feat)
        h_input = self.fc_cell_neigh(nodes.data['h_neigh1'])
        h = h_self + h_input


        return {'h': h}

    def apply_cell_func_level0(self, nodes):

        cell_feat = nodes.data['cell_feat']
        h = self.fc_cell_self(cell_feat)

        return {'h': h}




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
            graph.pull(cur_nodes,fn.copy_src('h','m'), fn.mean('m', 'h_neigh1'),
                       apply_node_func=self.apply_net_func, etype='net')

        else:
            if self.flag_attn:
                # calculate edge attention
                apply_node_func = self.apply_cell_func_level0 if level_id == 0 else self.apply_cell_func
                # apply_node_func = self.apply_cell_func
                reduce_func = fn.max('m', 'h_neigh1') if level_id == 0 else self.cell_msg_reduce_attn
                graph.pull(cur_nodes, self.message_func_attn, reduce_func=reduce_func,
                           apply_node_func=apply_node_func, etype='cell')
                graph.pull(cur_nodes, fn.copy_src('net_feat', 'm2'), reduce_func=fn.mean('m2', 'h_drive'),
                           apply_node_func=self.apply_netdrive_func, etype='net')
            else:
                apply_node_func = self.apply_cell_func_level0 if level_id == 0 else self.apply_cell_func
                #apply_node_func = self.apply_cell_func
                reduce_func = fn.max('m', 'h_neigh1') if level_id == 0 else self.cell_msg_reduce
                graph.pull(cur_nodes, fn.copy_src('h', 'm'), reduce_func=reduce_func,
                           apply_node_func=apply_node_func, etype='cell')

        # activation
        if self.activation is not None:
            graph.nodes['pin'].data['h'][cur_nodes] = self.activation(graph.nodes['pin'].data['h'][cur_nodes])
        # normalization
        if self.norm is not None:
            graph.nodes['pin'].data['h'][cur_nodes] = self.norm(graph.nodes['pin'].data['h'][cur_nodes])

        return graph.nodes['pin'].data['h'][targets]

        
class LayoutNet(nn.Module):
    def __init__(self,pooling):
        super(LayoutNet, self).__init__()
        activation = nn.ReLU()
        activation2 = nn.LeakyReLU(negative_slope=0.1)
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2,2,0)
        else:
            assert False, 'wrong pooling type for layoutnet!'
        self.encode = nn.Sequential(
            nn.Conv2d(2, 32, 9, 1, 4),
            activation,
            pooling_layer,
            # nn.ReLU(),
            nn.Conv2d(32, 64, 7, 1, 3),
            activation,
            pooling_layer,
            # nn.ReLU(),
            nn.Conv2d(64, 32, 9, 1, 4),
            activation,
            # nn.ConvTranspose2d(32, 16, 9, 2, 4, 1),
            # nn.Conv2d(16, 16, 5, 1, 2),
            # nn.ReLU(),
            nn.Conv2d(32, 1, 7, 1, 3),
            activation2
        )

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
        self, gnn,cnn,fcn,mlp_impact,mlp_weight,mlp_fuse,global_dim=32
    ):
        super(PathModel, self).__init__()
        self.global_dim=global_dim
        self.gnn = gnn
        self.cnn = cnn
        self.mlp_impact = mlp_impact
        self.mlp_weight = mlp_weight
        self.fcn = fcn
        self.mlp_fuse = mlp_fuse
        self.mlp_alpha = MLP(1,global_dim*2,global_dim)

    def forward(self, graph,nodes,eids,target_list,level_id,level_id_th,path_map):

        if self.fcn is not None and len(target_list) != 0:
            h_cnn =  self.fcn(path_map)
        else:
            h_cnn = None

        h_gnn = self.gnn(graph,nodes,eids,target_list,level_id) \
                   if self.gnn is not None \
                   else None

        h_global = self.mlp_alpha(level_id_th).expand(len(target_list),32)

        if len(target_list)==0:
           return None

        if h_cnn is None:
            h = th.cat([h_gnn,h_global],dim=1)
        elif h_gnn is None:
            h = th.cat([h_cnn,h_global],dim=1)
        else:
            h = th.cat((h_gnn,h_cnn,h_global),1)

        return self.mlp_fuse(h).squeeze(-1)

        return h
