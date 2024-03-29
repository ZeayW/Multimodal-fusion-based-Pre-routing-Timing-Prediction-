from importlib.resources import path
import sys

sys.path.append("./")

import dgl
from dgl.data import DGLDataset
import torch as th
import os
import networkx as nx
import dgl.function as fn
import json
from verilog_parser_asap7 import Parser


with open('../rawdata/cell_info_map.json', 'r') as f:
    cell_info_map = json.load(f)
with open('../rawdata/ctype2id.json', 'r') as f:
    ctype2id = json.load(f)
ctype2id['SRAM'] = len(ctype2id)
num_ctypes = len(ctype2id)


# ctype2id = {
#     "AND": 0, 
#     "FA": 1, 
#     "HA": 2, 
#     "MAJI": 3, 
#     "MAJ": 4, 
#     "NAND": 5, 
#     "NOR": 6, 
#     "OR": 7, 
#     "XNOR": 8, 
#     "XOR": 9, 
#     "BUF": 10, 
#     "CKINVDC": 11, 
#     "HB": 12, 
#     "INV": 13, 
#     "O2A1O1I": 14, 
#     "OA": 15, 
#     "OAI": 16, 
#     "A2O1A1I": 17, 
#     "A2O1A1O1I": 18, 
#     "AO": 19, 
#     "AOI": 20}
# num_ctypes = len(ctype2id)

def parse_single_file(parser,data_dir):
    r"""

    generate the DAG for a circuit design

    :param parser: DCParser
        the parser used to transform neetlist to DAG
    :param netlist_dir: str
        directory of the netlist of the design
    :param preopt_report_dir: str
        directory of the pre-optimization timing report
    :param postopt_report_dir: str
        directory of the post-optimization timing report
    :return: dglGraph
        the result DAG
    """


    # parse the netlist and repport files
    nx_graph,topo_levels,timing_paths,path_masks, PIs,pin2outcap,pin2delay, pin2trans = parser.parse(data_dir)
    
    if len(topo_levels)==0:
        return None,None,None,None,None,None
    
    print('--- Transforming to dgl graph...')
    print('\tassign nid to each node')
    # assign an id to each node
    node2id = {}
    for nd in nx_graph.nodes():
        if node2id.get(nd) is None:
            nid = len(node2id)
            node2id[nd] = nid

    # transform the topo_levels to fit dgl_graph
    dgl_topo_levels = []

    

    print('\t init the node features and labels...')
    # init the feature tensors
    is_start = th.zeros((len(node2id), 1), dtype=th.long)       # decide whether a node is the start point of one timing path
    is_end = th.zeros((len(node2id), 1), dtype=th.long)         # decide whether a node is the end point of one timing path
    is_critical = th.zeros((len(node2id), 1), dtype=th.long)    # decide whether an end point is critical
    arrival_time = th.zeros((len(node2id),1),dtype=th.float)
    required_time = th.zeros((len(node2id),1),dtype=th.float)
    # cell_feat is added to the output pin of each cell
    # the first num_ctypes bits of cell_feat is one-hot cell type vector,
    # and followed are cell load, area, width, height
    cell_feat = th.zeros((len(node2id), num_ctypes+8), dtype=th.float)
    # net_feaat is added to the sink pins of each net,
    # the first two bits of net_feat is the distance between the sink pin and the drive pin
    net_feat = th.zeros((len(node2id), 3), dtype=th.float)

    #paths = []
    #endpoints = []
    #endpoint2path = {}
    critical_paths = []
    #critical_ends = []
    path2endpoint = {}
    # collect the label information
    for i,path_info in enumerate(timing_paths):
        is_start[node2id[path_info.start]][0] = 1
        is_end[node2id[path_info.end]][0] = 1
        #arrival_time[]
        #paths.append(i)
        #endpoints.append(node2id[path_info.end])
        #endpoint2path[node2id[path_info.end]] = i
        path2endpoint[i] = node2id[path_info.end]
        arrival_time[node2id[path_info.end]][0] = path_info.arrival_time
        required_time[node2id[path_info.end]][0] = path_info.required_time
        if path_info.is_critical:
            is_critical[node2id[path_info.end]][0] = 1
            slack = path_info.required_time - path_info.arrival_time
            assert slack<0, 'critical path with positive slack!'
            critical_paths.append(i)
    path2level = {}
    for i,(level_nodes,targets,path_ids) in enumerate(topo_levels):
        dgl_topo_levels.append(
            ([node2id[nd] for nd in level_nodes],
             [node2id[nd] for nd in targets],
             path_ids
             ))
        for path_id in path_ids:
            path2level[path_id] = i

    #assert len(path2level) == len(paths)

    print('\ttransforming the edges...')

    # two types of edge: net / cell
    src_nodes = {'net':[],'cell':[]}
    dst_nodes = {'net':[],'cell':[]}
    edict_type = nx.get_edge_attributes(nx_graph,'edge_type')
    cdict_type = nx.get_node_attributes(nx_graph, 'cell_type')
    cdict_pos =  nx.get_node_attributes(nx_graph, 'position')
    cdict_port =  nx.get_node_attributes(nx_graph, 'port')
    #print(set(cdict_type.values()))
    #print(pin2trans)
    for pi in PIs:
        cell_name = cdict_type[pi]
        if cell_name == 'PI':
            continue
        else:
            cell_info = cell_info_map[cell_name]
            port_info = cell_info['pin_info'][cdict_port[pi]]
            total_outputcap = pin2outcap[pi]
            nid = node2id[pi]
            #type_id = cell_info['type_id']
            cell_type = cell_info['type']
            type_id = ctype2id[cell_type]
            cell_feat[nid][type_id] = 1                                     # cell type
            cell_feat[nid][num_ctypes] = cell_info['load']                # cell load
            cap = port_info['max_capacitance']
            if cell_name.startswith('SRAM') and cap=='':
                cap = '46.08'
            cell_feat[nid][num_ctypes + 1] = float(cap)                     # pin capacitance
            # cell_feat[nid][num_ctypes + 2] += src_port_info['capacitance']
            # cell_feat[nid][num_ctypes + 3] = max(cell_feat[nid][num_ctypes + 3],src_port_info['capacitance'])
            if cell_name.startswith('ICG') or cell_name.startswith('DHL') or cell_name.startswith('DLL') or (pin2trans.get(pi,None) is None and '/' not in pi):
                trans = 4
                delay = 0
            else:
                trans = pin2trans[pi]
                delay = pin2delay[pi]
            cell_feat[nid][num_ctypes + 2] = trans
            cell_feat[nid][num_ctypes + 3] = delay
            cell_feat[nid][num_ctypes + 4] = total_outputcap
            cell_feat[nid][num_ctypes + 5] = float(cell_info['area'])       # cell area
            cell_feat[nid][num_ctypes + 6] = float(cell_info['width'])      # cell width
            cell_feat[nid][num_ctypes + 7] = float(cell_info['height'])     # cell height
    #exit()
    cell2trans,cell2delay = {},{}
    ctype2trans,ctype2delay = {},{}
    for src, dst in nx_graph.edges():
        type = edict_type[(src, dst)]
        if type=='net': continue
        if pin2trans.get(dst,None) is not None:
            cell_name = cdict_type[dst]
            trans = pin2trans[dst]
            delay = pin2delay[dst]
            cell2trans[cell_name] = min(cell2trans.get(cell_name,trans),trans)
            cell2delay[cell_name] = min(cell2delay.get(cell_name,delay),delay)
            ctype = cell_info_map[cdict_type[dst]]['type']
            ctype2trans[ctype] = min(ctype2trans.get(ctype,trans),trans)
            ctype2delay[ctype] = min(ctype2delay.get(ctype,delay),delay)
    for src, dst in nx_graph.edges():
        type = edict_type[(src, dst)]
        assert type in ['cell','net'], 'Wrong edge type: {}'.format(type)
        src_nodes[type].append(node2id[src])
        dst_nodes[type].append(node2id[dst])

        # add the cell feature to the output pin of each cell
        dst_cell_info = cell_info_map[cdict_type[dst]]
        dst_port_info = dst_cell_info['pin_info'][cdict_port[dst]]
        #src_port_info = dst_cell_info['pin_info'][cdict_port[src]]
        if type== 'cell':
            cell_name = cdict_type[dst]
            nid = node2id[dst]
            total_outputcap = pin2outcap[dst]
            #type_id = dst_cell_info['type_id']
            cell_type = dst_cell_info['type']
            type_id = ctype2id[cell_type]
            cell_feat[nid][type_id] = 1                                     # cell type
            cell_feat[nid][num_ctypes] = dst_cell_info['load']                # cell load
            cap = dst_port_info['max_capacitance']
            #cap2 = cell_info['max_capacitance']
            #cap = cap2 if len(cap1)==0 else cap1
            #cell_feat[nid][num_ctypes + 1] = total_outputcap
            #assert dst not in PIs, '{} {} {}'.format(dst,cell_type,dst in PIs)
            #print('addr[0]' not in PIs)
            if cell_name.startswith('SRAM') and cap=='':
                cap = '46.08'
            elif cap == '':
                print(cell_name,cdict_port[dst],cap)
            cell_feat[nid][num_ctypes + 1] = float(cap)                     # max output pin capacitance
            
            # cell_feat[nid][num_ctypes + 2] += src_port_info['capacitance']
            # cell_feat[nid][num_ctypes + 3] = max(cell_feat[nid][num_ctypes + 3],src_port_info['capacitance'])

            if cell_name.startswith('ICG'):
                trans = 4
                delay = 0
            else:
                if pin2trans.get(dst,None) is None:
                    if ctype2trans.get(cell_type,None) is None:
                        print(dst, cell_type)
                    trans = cell2trans.get(cell_name,ctype2trans.get(cell_type,0))
                    delay = cell2delay.get(cell_name,ctype2delay.get(cell_type,0))
                else:
                    trans = pin2trans[dst]
                    delay = pin2delay[dst]
                
            cell_feat[nid][num_ctypes + 2] = trans
            cell_feat[nid][num_ctypes + 3] = delay
            #cell_feat[nid][num_ctypes + 2] = pin2trans[dst]
            #cell_feat[nid][num_ctypes + 3] = pin2delay[dst]
            cell_feat[nid][num_ctypes + 4] = total_outputcap
            cell_feat[nid][num_ctypes + 5] = float(dst_cell_info['area'])       # cell area
            cell_feat[nid][num_ctypes + 6] = float(dst_cell_info['width'])      # cell width
            cell_feat[nid][num_ctypes + 7] = float(dst_cell_info['height'])
        # add the net feature to the sink pins of each net
        elif type == 'net':
            nid = node2id[dst]
            distance = (abs(cdict_pos[dst][0]-cdict_pos[src][0]),
                        abs(cdict_pos[dst][1]-cdict_pos[src][1]))
            net_feat[nid][0] = distance[0]      # x-axis distance
            net_feat[nid][1] = distance[1]      # y-axis distance
            if cdict_port.get(src,None) in (None, 'PI'):
                drive_cap = '0'
            else:
                drive_cell_info = cell_info_map[cdict_type[src]]
                drive_port_info = drive_cell_info['pin_info'][cdict_port[src]]
                drive_cap = drive_port_info['max_capacitance']
                if cdict_type[src].startswith('SRAM') and drive_cap == '': drive_cap = '46.08'
            #net_feat[nid][2] = float(drive_cap)
            total_sink_cap = pin2outcap[src]
            #net_feat[nid][3] = total_sink_cap
            cap = dst_port_info['capacitance']
            cap = '13.0' if len(cap)==0 else cap
            net_feat[nid][2] = float(cap)
        #ntype[node2id[dst]][0] = 0 if type == 'cell' else 1

    # build the dgl heterograph
    #   two types of edge:
    #       net / cell
    print('\t build the dgl graph...')
    graph = dgl.heterograph(
        {('pin','net','pin'):(th.tensor(src_nodes['net']), th.tensor(dst_nodes['net'])),
         ('pin','cell','pin'):(th.tensor(src_nodes['cell']), th.tensor(dst_nodes['cell']))
         }
    )
    

    graph.nodes['pin'].data['start'] = is_start
    graph.nodes['pin'].data['end'] = is_end
    graph.nodes['pin'].data['label'] = is_critical
    graph.nodes['pin'].data['arrival_time'] = arrival_time
    graph.nodes['pin'].data['required_time'] = required_time
    graph.nodes['pin'].data['cell_feat'] = cell_feat
    graph.nodes['pin'].data['net_feat'] = net_feat

    
    print('--- Transforming is done!')
    print('Processing is Accomplished!')
    #print('\t',graph)
    nodes = th.tensor(range(graph.number_of_nodes()))

    print('\tnum start: ', len(nodes[graph.nodes['pin'].data['start'].squeeze(1)==1]))
    print('\tnum end: ', len(nodes[graph.nodes['pin'].data['end'].squeeze(1) == 1]))
    print('\tnum critical end: ', len(nodes[graph.nodes['pin'].data['label'].squeeze(1) == 1]))

    return graph,dgl_topo_levels,path_masks,path2level,path2endpoint,critical_paths

class Dataset(DGLDataset):
    def __init__(self, top_module,masking,data_dir):
        self.data_dir = data_dir
        print(top_module)
        self.parser = Parser(top_module,masking)
        super(Dataset, self).__init__(name="dac22")

    def process(self):
        r"""

        transform the netlists to DAGs

        :return:

        """
        self.len = 1

        # parse the data
        graph,topo_levels,path_masks,path2level,path2endpoint,critical_paths = parse_single_file(
            self.parser,self.data_dir)
        
        self.graph = graph               # dgl graph representation of the netlist
        self.topo_levels = topo_levels   # topological level of the graph
        self.path_masks  = path_masks
        #self.paths = paths
        self.path2level = path2level
        self.path2endpoint = path2endpoint
        self.critical_paths = critical_paths
        
    def __len__(self):
        return self.len




def main():
    postopt_report_path = '../data/tinyrocket/post_opt/path.tarpt'
    preopt_report_path = '../data/tinyrocket/pre_opt/path.tarpt'
    netlist_path = '../data/tinyrocket/netlist.v'
    cell_loc_file = '../data/tinyrocket/cell_loc.txt'
    parser = Parser(top_module='ChipTop')
    parse_single_file(parser,netlist_path,preopt_report_path,postopt_report_path,cell_loc_file)


if __name__ == "__main__":
    seed = 1234
    main()

