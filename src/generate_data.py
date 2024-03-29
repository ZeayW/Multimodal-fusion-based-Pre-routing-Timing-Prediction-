from dataset import *
from options import get_options
import pickle
import os
import torch as th

top_map = {
    'darkriscv': 'darkriscv',
    'sha3': 'ChipTop',
    'smallboom': 'BoomCore',
    'rocket': 'ChipTop',
    'xgate': 'xgate_top',
    'ae18': 'ae18_core',
    'or1200': 'or1200_top',
    'hwacha':'Hwacha',
    'steelcore': 'steel_core_top',
    'tinyrocket':"ChipTop",
    'chacha':'chacha',
    'arm9':'arm9_compatiable_code',
    'r8051':'r8051',
    'jpeg':'jpeg_top',

}

options = get_options()

rawdata_path = options.rawdata_path
data_save_path = options.data_save_path


# parse the raw_data listed in the txt file

os.makedirs(data_save_path,exist_ok=True)
for design in os.listdir(rawdata_path):
    #if design!='chacha': continue
    if design in ('util.py','late_lib.json','early_lib.json','README.txt','def','run.sh','ae18','steel-core'):
        continue
    print('-------- Parsing design: {}...'.format(design))
    design_save_path = os.path.join(data_save_path,'{}.pkl'.format(design))
    if os.path.exists(design_save_path):
        print('Design {} already parsed! Skip'.format(design))
        continue
    design_dir = os.path.join(rawdata_path,design)
    top_module = top_map[design]
    th.multiprocessing.set_sharing_strategy('file_system')
    dataset = Dataset(top_module, options.masking,design_dir)
    with open(os.path.join(design_dir,'features/datas.pkl'),'rb') as f:
        cnn_inputs = pickle.load(f)
    if dataset.graph is not None:
        th.save(
            (dataset.graph,dataset.topo_levels,dataset.path_masks,
                dataset.path2level,dataset.path2endpoint,dataset.critical_paths,cnn_inputs) 
            ,design_save_path
        )
  
