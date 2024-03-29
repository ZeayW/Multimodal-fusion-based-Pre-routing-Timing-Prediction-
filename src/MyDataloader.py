from dgl import subgraph
from torch.utils.data import DataLoader,Dataset

def sample_topo_levels(graph,topo_levels,sampled_path_ids,sampled_starts,sampled_targets):
    
     #num_levels = len(self.reverse_topo_levels)
    sampled_topo_levels = []
    pre_nodes = []
    flag_first = True
    num_levels = len(topo_levels)
    PIs = sampled_starts
    sampled_topo_levels.append(PIs)
    #pre_nodes = PIs
    # for level_id, (nodes,targets,path_ids) in topo_levels[1:]:
    #     remain_targets = sampled_targets.get(level_id,[])
    #     remain_path_ids = sampled_path_ids.get(level_id,[])

    #     edge_type = 'net ' if level_id%2==0 else 'cell'
    #     remain_nodes =  graph.out_edges(pre_nodes,etype=edge_type)[1].cpu().numpy().tolist()
    #     remain_nodes = set(remain_nodes).intersection(set(nodes))
    #     if level_id%2==0:
    #         remain_nodes = 

    pre_nodes = []
    all_remain_nodes = []
    for level_id in range(num_levels-1,-1,-1):
        nodes = set(topo_levels[level_id])
        remain_targets = sampled_targets.get(level_id,[])
        remain_path_ids = sampled_path_ids.get(level_id,[])

        if len(remain_targets)==0 and flag_first:
            continue 
        
        all_remain_nodes.extend(graph.in_edges(pre_nodes,)[0].cpu().numpy().tolist())
        all_remain_nodes.extend(remain_targets)

        cur_remain_nodes = nodes.intersection(all_remain_nodes)

        pre_nodes = cur_remain_nodes

        edge_type = 'net' if level_id%2==0 else 'cell'
        if flag_first:
            remain_nodes = remain_targets
        else:
            remain_nodes = subgraph.in_subgraph(graph,pre_nodes).edges[edge_type][0]['_ID'].cpu().numpy().tolist()
            remain_nodes.extend(remain_targets)
            print(pre_nodes)
            print(subgraph.in_subgraph(graph,pre_nodes))
            # print('cell feat',graph.ndata['cell_feat'][pre_nodes])
            # print('net feat',graph.ndata['net_feat'][pre_nodes])
            print(level_id,len(remain_targets),len(remain_nodes))
            print('net feat',graph.ndata['net_feat'][remain_nodes])
        flag_first = False
        pre_nodes = remain_nodes
        sampled_topo_levels.append((remain_nodes,remain_targets,remain_path_ids))
    sampled_topo_levels.reverse()
    for level_id,(nodes,targets,path_ids) in enumerate(sampled_topo_levels):
        print('level :',level_id,len(nodes),len(targets))
    return sampled_topo_levels


class PathDataset(Dataset):
    def __init__(self, paths):
        
        self.paths = paths


    def __getitem__(self, index):
       
        return self.paths[index]

    def __len__(self):
        return len(self.paths)
    
    
