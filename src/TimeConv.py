"""Torch Module for PathConv layer"""

import torch as th
from torch import nn
from dgl import function as fn
from dgl.nn.functional import edge_softmax


def cell_msg_reduce(nodes):
    return {'h_neigh1': th.max(nodes.mailbox['m'], dim=1)}


class MLP(nn.Module):
    r"""
                Description
                -----------
                a simple multilayer perceptron
    """
    def __init__(self,in_dim,out_dim,bias,activation =nn.ReLU() ,dropout=0.5):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        hidden_dim = 256

        self.layers= nn.Sequential(
            nn.Linear(in_dim,hidden_dim,bias=bias),
            self.activation,
            nn.Linear(hidden_dim,out_dim,bias=bias),
        )
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.layers[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.layers[2].weight, gain=gain)

    def forward(self,embedding):
        return self.layers(embedding).squeeze(-1)


class PathConv(nn.Module):

    def __init__(self,
                 in_feats_dim,
                 out_feats_dim,
                 cell_feat_dim,
                 net_feat_dim,
                 flag_attn=False,
                 num_heads = 1,
                 activation=th.nn.functional.relu,
                 #activation = th.nn.LeakyReLU(),
                 bias=True,
                 norm=None):
        super(PathConv, self).__init__()
        self.flag_attn = flag_attn
        self.in_feat_dim = in_feats_dim
        self.out_feats_dim = out_feats_dim
        self.cell_feat_dim = cell_feat_dim
        self.net_feat_dim = net_feat_dim
        self.num_heads = num_heads
        # initialize the gate functions, each for one gate type, e.g., AND, OR, XOR...
        self.fc_cell_neigh = MLP(self.in_feat_dim, self.out_feats_dim, bias=bias)
        self.fc_cell_self = MLP(self.cell_feat_dim, self.out_feats_dim, bias=bias)
        #self.fc_net_neigh = MLP(self.in_feat_dim, self.out_feats_dim, bias=bias)
        self.fc_net_self = MLP(self.net_feat_dim, self.out_feats_dim, bias=bias)
        #self.fc_cell_neigh = nn.Linear(self.in_feat_dim, self.out_feats_dim, bias=bias)
        #self.fc_cell_self = nn.Linear(self.cell_feat_dim, self.out_feats_dim, bias=bias)
        #self.fc_net_neigh = nn.Linear(self.in_feat_dim, self.out_feats_dim, bias=bias)
        #self.fc_net_self = nn.Linear(self.net_feat_dim, self.out_feats_dim, bias=bias)

        negative_slope = 0.2
        if self.flag_attn:
            self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, self.out_feats_dim)))
        #self.leaky_relu = nn.LeakyReLU(negative_slope)

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
        #nn.init.xavier_uniform_(self.fc_net_neigh.weight, gain=gain)
        #nn.init.xavier_uniform_(self.fc_net_self.weight, gain=gain)
        #nn.init.xavier_uniform_(self.fc_cell_neigh.weight, gain=gain)
        #nn.init.xavier_uniform_(self.fc_cell_self.weight, gain=gain)
        if self.flag_attn:
            nn.init.xavier_normal_(self.attn, gain=gain)
       

    def apply_net_func(self,nodes):

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
        # h_net = f1(net_feat)+f2(h_drive)
        #print('h_neigh_net',nodes.data['h_neigh1'])
        # if len(nodes)==22283:
        #     # print('net feat',nodes.data['net_feat'])
        #     # print('h_neigh',nodes.data['h_neigh1'])
        #     print('fc_net_self',self.fc_net_self.weight)
        #     print('fc_net_neigh',self.fc_net_neigh.weight)
        h_self = self.fc_net_self(nodes.data['net_feat'])
        #h_drive = self.fc_net_neigh(nodes.data['h_neigh1'])
        h_drive = nodes.data['h_neigh1']
        h = h_self + h_drive
        return {'h':h}

    def cell_msg_reduce(self,nodes):
        # print(th.max(nodes.mailbox['m'], dim=1).values)
        # exit()
        msg = nodes.mailbox['m']
        weight = th.softmax(msg,dim=1)
        return{'h_neigh1':(msg*weight).sum(1)}

    def apply_cell_func(self,nodes):

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
        # h_cell = f1(cell_feat)+f2(max(h_input))
        #print(nodes.data['cell_feat'])
        h_self = self.fc_cell_self(nodes.data['cell_feat'])
        h_input = self.fc_cell_neigh(nodes.data['h_neigh1'])
        h = h_self + h_input
        #print('h',h)
        return {'h':h}

    def apply_cell_func_level0(self,nodes):
        h_self = self.fc_cell_self(nodes.data['cell_feat'])
        # print('h',h)
        return {'h': h_self}

    def copy_src(self,edges):
        return {'m':edges.src['h']}

    def cal_edge_attn(self,edges):
        e = self.activation(
                (edges.src['h'].view((-1,1,self.out_feats_dim))*self.attn).sum(dim=-1)
                )
        return {'e':e}

    def forward(self,graph,cur_nodes,eids,targets,level_id):
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
        #print(graph.nodes['pin'].data['h'].shape)
        #graph.srcdata['h'][pre_nodes] = input_feature
        #graph.ndata['h'] = graph.ndata['h'].detach()

        if level_id%2==1:
            graph.pull(cur_nodes, fn.copy_src('h', 'm'), fn.mean('m','h_neigh1'),
                       apply_node_func=self.apply_net_func,etype='net')

        else:
            if self.flag_attn and level_id!=0:
                # calculate edge attention
                graph.apply_edges(func=self.cal_edge_attn,edges=eids,etype='cell')
                # apply edge softmax
                # our version is based on sparse matrix softmax
                e= graph.edges['cell'].data['e'][eids]
                target_edges= (graph.edges(etype='cell')[0][eids],graph.edges(etype='cell')[1][eids])
                i = th.stack(target_edges, 0)
                sp = th.sparse.FloatTensor(i, e)
                a = th.sparse.softmax(sp, 0).coalesce().values().view((-1,1))
                graph.edges['cell'].data['a'][eids] = a
                # message passing
                graph.pull(cur_nodes, fn.u_mul_e('h', 'a', 'm'), fn.sum('m','h_neigh1'),
                       apply_node_func=self.apply_cell_func,etype='cell')
            else:
                # print(cur_nodes)
                # print(level_id)
                # print('#nodes',len(cur_nodes))
                apply_node_func = self.apply_cell_func_level0 if level_id==0 else self.apply_cell_func
                graph.pull(cur_nodes, fn.copy_src('h','m'), self.cell_msg_reduce,
                            apply_node_func=apply_node_func, etype='cell')

                # change the apply_func for PIs
                # graph.pull(cur_nodes, fn.copy_src('h','m'), fn.max('m','h_neigh1'),
                #             apply_node_func=apply_node_func, etype='cell')

                #orignal one
                # graph.pull(cur_nodes, fn.copy_src('h','m'), fn.max('m','h_neigh1'),
                #             apply_node_func=self.apply_cell_func, etype='cell')






        #activation
        #print(graph.nodes['pin'].data['h'][cur_nodes])
        if self.activation is not None:
            graph.nodes['pin'].data['h'][cur_nodes] = self.activation(graph.nodes['pin'].data['h'][cur_nodes])
        # normalization
        if self.norm is not None:
            graph.nodes['pin'].data['h'][cur_nodes] = self.norm(graph.nodes['pin'].data['h'][cur_nodes])
        #
        # if level_id==0:
        #     print(graph.nodes['pin'].data['h'][cur_nodes])
        #print(graph.nodes['pin'].data['h'][cur_nodes])
        return graph.nodes['pin'].data['h'][targets]
