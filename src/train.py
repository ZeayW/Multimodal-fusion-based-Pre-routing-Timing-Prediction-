r"""
this script is used to train and validate the models
"""
from torchmetrics import R2Score
import random
from importlib.resources import path
from lib2to3.pytree import Node
from tkinter import N
from tracemalloc import start
import json
from options import get_options
from model import *
from Unet import UNet
import dgl
import pickle
import numpy as np
import os
from time import time
from random import shuffle
import itertools
from MyDataloader import *
import tee
from torch.utils.data import DataLoader

with open('../rawdata/ctype2id.json', 'r') as f:
    ctype2id = json.load(f)
    num_ctypes = len(ctype2id)


device = th.device("cuda:"+str(get_options().gpu) if th.cuda.is_available() else "cpu")
R2_score = R2Score().to(device)
Loss = nn.CrossEntropyLoss() if get_options().task == 'cls' else nn.MSELoss() 

def init_model(options):
    r"""

    initialize the model

    :param options:
        some additional parameters
    :return:
        param: options
        gnn : initialized gnn
        cnn: initialized gnn
        mlp: initialized mlp
    """

    # initialize the GNN model
    print('Intializing models...')
    #print(options.no_cnn,options.no_gnn)
    assert not options.no_cnn or not options.no_gnn, 'GNN and CNN model can not be both None!'
    mlp_dim = 0
    if options.no_gnn:
        gnn = None
    else:
        gnn = PathConv(
            out_feat_dim=options.out_dim,
            hidden_feat_dim=options.out_dim,
            cell_feat_dim = options.cell_feat_dim,
            net_feat_dim = options.net_feat_dim,
            flag_attn= options.attn,
            num_heads= options.num_heads
        )
        mlp_dim += gnn.out_feat_dim
    # initialize the cnn model
    if options.no_cnn:
        cnn = None
        fcn = None
    else:
        cnn = UNet(options.pooling) if options.unet else LayoutNet(options.pooling)
        fcn = nn.Linear(options.map_size*options.map_size,options.cnn_outdim)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(fcn.weight, gain=gain)
        mlp_dim += options.cnn_outdim
    # initialze the MLP
    mlp_dim += 64
    mlp = MLP(mlp_dim,mlp_dim*2,options.nlabels)

   
    #fcn=None
    model = PathModel(gnn,fcn,mlp)
    print("creating model in:",options.model_saving_dir)
    print('Path Model',model)
    print('cnn:',cnn)
    # save the model and create a file to save the results
    if os.path.exists(os.path.join(options.model_saving_dir,'model.pkl')) is False:
        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
            parameters = options
            pickle.dump((parameters, model,cnn), f)
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'w') as f:
            pass
    print('Model Initializing is accomplished!')

    return parameters, model,cnn

def load_model(device,options):
    r"""
    Load the model

    :param device:
        the target device that the model is loaded on
    :param options:
        some additional parameters
    :return:
        param: new options
        gnn : loaded gnn
        cnn: loaded gnn
        mlp: loaded mlp
    """
    print('----------------Loading the model and hyper-parameters----------------')
    model_dir = options.model_saving_dir
    # if there is no model in the target directory, break
    if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
        param, model,cnn = init_model(options)       
    else:
        # read the pkl file that saves the hype-parameters and the model.
        with open(os.path.join(model_dir,'model.pkl'), 'rb') as f:
            # param: hyper-parameters, e.g., learning rate;
            #load the model from pickle file
            param, model,cnn = pickle.load(f)
            param.model_saving_dir = options.model_saving_dir
            # make some changes to the options
            if options.change_lr:
                param.learning_rate = options.learning_rate
            if options.change_alpha:
                param.alpha = options.alpha
    model = model.to(device)
    if cnn is not None: cnn = cnn.to(device)
    # model = nn.DataParallel(model, device_ids=[0,1])
    # cnn = nn.DataParallel(cnn, device_ids=[0,1])
    
    print('Model and hyper-parameters successfully loaded!')
    return param, model, cnn



def validate(data_save_path,val_designs,device,model,cnn,beta,options):
    r"""

    validate the model

    :param loader:
        the data loader to load the validation dataset
    :param device:
        device
    :param model:
        trained model
    :param mlp:
        trained mlp
    :param Loss:
        used loss function
    :param beta:
        a hyperparameter that determines the thredshold of binary classification
    :param options:
        some parameters
    :return:
        result of the validation: loss, acc,recall,precision,F1_score
    """

    overall_loss, overall_acc, overall_recall, overall_precision, overall_f1,  overall_r2 = 0, 0.0, 0, 0, 0, 0
    # runtime = 0
    res = []
    with th.no_grad():
        # load validation data, one batch at a time
        # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
        # and output neighborhoods (out_block).
        # The dst_nodes of the last block of in_block/out_block is the central nodes.
        print('validate:')
        case_idx = 0
        for i, design in enumerate(val_designs):
            path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks = load_single_design(
                'test',data_save_path,design,options.out_dim,options.os_rate,options.feat_reduce,options.norm)

            total_num, total_loss, correct, fn, fp, tn, tp, total_r2 = 0, 0.0, 0,0, 0, 0, 0, 0
            runtime = 0
            # optim.zero_grad()
            start_time = time()
            cnn_inputs = cnn_inputs.reshape((1,cnn_inputs.shape[0],cnn_inputs.shape[1],cnn_inputs.shape[2]))
            feat_map = cnn(cnn_inputs.to(device)).reshape((1,-1)) if cnn is not None else None
            #path_mask = path_mask.to_sparse()
            # transfer the data to GPU

            graph = graph.to(device)
            count_target = 0
            label_hats = None
            target_list = []
            # first_level_nodes = topo_levels[0][0]
            # init_message = graph.ndata['h'][first_level_nodes]
            path_loader = DataLoader(path_dataset,batch_size=len(path_dataset.paths),shuffle=False)
            for path_ids in path_loader:
                path_ids = list(set(path_ids.numpy().tolist()))
                sampled_ends, sampled_paths = {}, {}
                for i , pathid in enumerate(path_ids):
                    level = path2level[pathid]
                    endpoint = path2endpoint[pathid]
                    #assert endpoints2path[endpoint] == pathid
                    sampled_ends[level] = sampled_ends.get(level,[])
                    sampled_ends[level].append(endpoint)
                    sampled_paths[level] = sampled_paths.get(level,[])
                    sampled_paths[level].append(pathid)
                
                for level_id, level in enumerate(topo_levels):
                    nodes,eids = level[:2]
                    if len(level)==4: eids = eids.to(device)
                    targets = sampled_ends.get(level_id,[])
                    paths = sampled_paths.get(level_id,[])
                    target_list.extend(targets)
                    count_target += len(target_list)

                    if options.no_cnn or len(paths) == 0:
                        path_map = None
                    else:
                        path_mask = th.index_select(path_masks,0,th.tensor(paths)).to(device)
                        path_map = path_mask.to_dense()*feat_map
                        #path_map = path_map.view(-1,path_map.shape[1]*path_map.shape[2])
                
                    cur_label_hats = model(graph,nodes,eids,targets,level_id,th.tensor(level_id,dtype=th.float).unsqueeze(0).to(device),path_map)

                    if len(paths) == 0:
                        continue

                    if label_hats is None:
                        label_hats = cur_label_hats
                    else:
                        label_hats = th.cat((label_hats,cur_label_hats),dim=0)
                    
            # labels: the ground-truth binary labels, decide whether path is critical or not
            # predict_labels: the predicted binary labels
            # label_hats: the output of the model, len=2 for classification, len=1 for regression
            labels = graph.ndata['label'][target_list].squeeze()
            if options.task == 'cls':
                predict_labels = th.argmax(nn.functional.softmax(label_hats, 1), dim=1)
                test_loss = Loss(label_hats, labels)
                test_r2 = 0                
            elif options.task == 'reg':
                required_time = graph.ndata['required_time'][target_list].squeeze()
                arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
                test_loss = Loss(label_hats, arrival_time)
                predict_labels = judge_critical(label_hats,required_time).to(device)
                test_r2 = R2_score(label_hats,arrival_time).to(device)
                total_r2 += test_r2.item() 
            #print('R2 score: {}'.format(test_r2))
            # calculate loss

            #print(cnn.down3.maxpool_conv.weights)
            total_num += len(labels)
            total_loss += test_loss.item()
            

            correct += (
                    predict_labels == labels
            ).sum().item()
            # calculate fake negative, true positive, fake negative, and true negative rate
            fn += ((predict_labels == 0) & (labels != 0)).sum().item()
            tp += ((predict_labels != 0) & (labels != 0)).sum().item()
            tn += ((predict_labels == 0) & (labels == 0)).sum().item()
            fp += ((predict_labels != 0) & (labels == 0)).sum().item()

            acc = correct / total_num
            recall = 0
            precision = 0
            if tp != 0:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
            F1_score = 0
            if precision != 0 or recall != 0:
                F1_score = 2 * recall * precision / (recall + precision)

            overall_loss += total_loss
            overall_r2 += total_r2
            overall_recall += recall
            overall_f1 += F1_score
            overall_acc += acc
            overall_precision += precision

            print("\tcase {} \tl:{:.3f}, r2:{:.3f}, rc:{:.3f}, F1:{:.3f}".format(case_idx,test_loss, test_r2,recall, F1_score))
            case_idx +=1
            res.append([test_loss,test_r2,acc,recall,precision,F1_score])
    # calculate the overall loss / accuracy
    num_case = case_idx
    overall_loss = overall_loss / num_case
    overall_acc = overall_acc / num_case
    overall_r2 = overall_r2 / num_case
    overall_f1 = overall_f1 / num_case
    overall_recall = overall_recall / num_case
    overall_precision = overall_precision / num_case
    # calculate overall recall, precision and F1-score

    print("\toverall r2:{:.3f}, rc:{:.3f}, F1:{:.3f}".format(overall_r2, overall_recall, overall_f1))
    
    return res,overall_f1,overall_r2


def split_dataset(paths,critical_paths):
    non_critical_paths = list(set(paths)-set(critical_paths))
            
    shuffle(critical_paths)
    val_paths = critical_paths[:int(len(critical_paths)/5)]
    test_paths = critical_paths[int(len(critical_paths)/5):]
    shuffle(non_critical_paths)
    val_paths.extend(non_critical_paths[:int(len(non_critical_paths)/5)])
    test_paths.extend(non_critical_paths[int(len(non_critical_paths)/5):])

    return val_paths,test_paths




def minMax_scalar(a):
    min_a = th.min(a)
    max_a = th.max(a)
    return (a - min_a) / (max_a - min_a)

def norm(feature,start_idx):
    num_feat = feature.shape[1]
    for i in range(start_idx,num_feat):
        feature[:,i:i+1] = minMax_scalar(feature[:,i]).reshape(-1,1)
    return feature


def get_design_list(data_path,usage):
    assert usage in ['train', 'test'], "Wrong data usage! Should be either 'train' or 'test'."
    design_list_file = os.path.join(data_path, '{}data_list.txt'.format(usage))
    assert os.path.exists(design_list_file), \
        "Can not find the traindata list txt '{}'".format(design_list_file)

    with open(design_list_file, 'r') as f:
        lines = f.readlines()
        design_list = [l.replace('\n', '') for l in lines]
    print('--- {} designs: '.format(usage), design_list)


    return design_list

def load_single_design(usage,data_path,design,init_feat_dim,os_rate,feat_reduce,if_norm):
    dataset_file = os.path.join(data_path, '{}.pkl'.format(design))
    graph, topo_levels, path_masks, path2level, path2endpoint, critical_paths, cnn_inputs = th.load(dataset_file)
    # with open(dataset_file,'rb') as f:
    # graph,topo_levels,path_masks,path2level,path2endpoint,critical_paths,cnn_inputs = pickle.load(f)
    # print(path_masks.shape, graph.ndata['cell_feat'].shape)

    graph.ndata['h'] = th.zeros((graph.number_of_nodes(), init_feat_dim), dtype=th.float)
    graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float)
    if feat_reduce is not None:
        if feat_reduce[1] != 0:
            graph.ndata['net_feat'] = graph.ndata['net_feat'][:, :-feat_reduce[1]]
        if feat_reduce[0] != 0:
            graph.ndata['cell_feat'] = graph.ndata['cell_feat'][:, :-feat_reduce[0]]

    if if_norm:
        graph.ndata['cell_feat'] = norm(graph.ndata['cell_feat'], num_ctypes)
        graph.ndata['net_feat'] = norm(graph.ndata['net_feat'], num_ctypes)

    if type(cnn_inputs) == np.ndarray:
        cnn_inputs = th.from_numpy(cnn_inputs).float()
    # cnn_inputs = th.unsqueeze(cnn_inputs,dim=0)
    paths = list(range(len(graph.ndata['end'][graph.ndata['end'].squeeze() == 1])))
    # non_critical_paths = list(set(paths)-set(critical_paths))
    num_neg = len(paths) - len(critical_paths)
    num_pos = len(critical_paths)
    ratio = num_neg / num_pos - 1

    if usage == 'test':

        split_file = os.path.join(data_path, '{}_split.pkl'.format(design))
        if os.path.exists(split_file):
            with open(split_file, 'rb') as f:
                val_paths, test_paths = pickle.load(f)

        else:
            val_paths, test_paths = split_dataset(paths, critical_paths)
            with open(split_file, 'wb') as f:
                pickle.dump((val_paths, test_paths), f)
        paths = val_paths
        # print(len(paths), len(critical_paths))

    if usage == 'train' and os_rate != 0 and ratio > 1:
        # shuffle(critical_paths)
        for _ in range(os_rate):
            paths.extend(critical_paths)
        # while ratio>=1:
        #     paths.extend(critical_paths)
        #     ratio -= 1
        # shuffle(critical_paths)
        # paths.extend(critical_paths[:int(ratio*num_pos)])
    path_dataset = PathDataset(paths)

    return path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks


def judge_critical(pred_arr_time,required_time):
    pred_slack = required_time - pred_arr_time
    is_critical = th.ones(pred_arr_time.shape)
    is_critical[pred_slack>=0] = 0
    return is_critical

def train(options,seed):

    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:"+str(options.gpu) if th.cuda.is_available() else "cpu")

    # you can define your dataset file here
    data_save_path = options.data_save_path
    print(data_save_path)

    # load the model
    options.cell_feat_dim -= options.feat_reduce[0]
    options.net_feat_dim -= options.feat_reduce[1]
    options, model,cnn = load_model(device, options)
    with open(os.path.join(options.model_saving_dir, 'seed.txt'), 'a') as f:
        f.write(str(seed))
    
    print('Hyperparameters are listed as follows:')
    print(options)
    print('seed:',seed)
    print('The model architecture is shown as follow:')
    print(model)
    print(cnn)

    print("----------------Loading data----------------")
    #train_data_file = os.path.join(data_save_path, 'train.pkl')

    train_designs = get_design_list(data_save_path,'train')
    val_designs = get_design_list(data_save_path,'test')

    # split the validation set and test set

    beta = options.beta
    # set the optimizer
    if cnn is not None:         
        optim = th.optim.Adam(
            itertools.chain(model.parameters(), cnn.parameters()),
            #model.parameters(),
            options.learning_rate, weight_decay=options.weight_decay
        )
        cnn.train()
        model.train()
    else:
        optim = th.optim.Adam(
            model.parameters(),
            options.learning_rate, weight_decay=options.weight_decay
        )
        model.train()
    
    
    print("----------------Start training---------------")
    pre_loss = 100
    stop_score = 0
    max_F1_score, max_r2 = 0, 0


    th.autograd.set_detect_anomaly(True)
    for epoch in range(options.num_epoch):
        runtime = 0

        total_num,total_loss,correct,fn,fp,tn,tp,total_r2 = 0,0.0,0,0,0,0,0,0

        #total_nodes = []
        
        #shuffle(train_dataset)
        for i, design in enumerate(train_designs):
            path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks = load_single_design(
                'train',data_save_path,design,options.out_dim,options.os_rate,options.feat_reduce,options.norm)

            feat_map = cnn(cnn_inputs.to(device)).reshape((1,-1)) if cnn is not None else None

            graph = graph.to(device)
            #print(graph)
            if len(path2level)<= options.batch_size:
                path_loader = DataLoader(path_dataset,batch_size=options.batch_size,shuffle=True,drop_last=False)
            else:
                path_loader = DataLoader(path_dataset,batch_size=options.batch_size,shuffle=True,drop_last=True)

            num_batch = len(path_loader)
            for bidx,path_ids in enumerate(path_loader):
                path_ids = list(path_ids.numpy().tolist())
                sampled_ends, sampled_paths = {}, {}
                for i , pathid in enumerate(path_ids):
                    level = path2level[pathid]
                    endpoint = path2endpoint[pathid]
                    sampled_ends[level] = sampled_ends.get(level,[])
                    sampled_ends[level].append(endpoint)
                    sampled_paths[level] = sampled_paths.get(level,[])
                    sampled_paths[level].append(pathid)

                count_target = 0
                label_hats = None
                target_list = []
                # please start from the first level!!!!
                for level_id, level in enumerate(topo_levels):
                    nodes,eids = level[:2]
                    if len(level)==2: eids = eids.to(device)
                    targets = sampled_ends.get(level_id,[])
                    paths = sampled_paths.get(level_id,[])
                    target_list.extend(targets)
                    count_target += len(target_list)
                    if options.no_cnn or len(paths) == 0:
                        path_map = None                   
                    else:
                        path_mask = th.index_select(path_masks,0,th.tensor(paths)).to(device)
                        path_map = path_mask.to_dense()*feat_map

                    cur_label_hats = model(graph,nodes,eids, targets,level_id,th.tensor(level_id,dtype=th.float).unsqueeze(0).to(device),path_map)

                    if len(paths) == 0:
                        continue

                    if label_hats is None:
                        label_hats = cur_label_hats
                    else:
                        label_hats = th.cat((label_hats,cur_label_hats),dim=0)

                labels = graph.ndata['label'][target_list].squeeze()

                if options.task == 'cls':
                    predict_labels = th.argmax(nn.functional.softmax(label_hats, 1), dim=1)
                    train_loss = Loss(label_hats, labels)
                    train_r2 = 0
                elif options.task == 'reg':
                    required_time = graph.ndata['required_time'][target_list].squeeze()
                    arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
                    train_loss = Loss(label_hats, arrival_time)
                    predict_labels = judge_critical(label_hats,required_time).to(device)
                    train_r2 = R2_score(label_hats,arrival_time).to(device)
                    total_r2 = train_r2.item() * len(labels)
                # calculate loss
                total_num = len(labels)
                total_loss = train_loss.item() * len(labels)

                # calculate accuracy
                correct = (
                        predict_labels == labels
                ).sum().item()
                
                # calculate accuracy, recall, precision and F1-score
                acc = correct / total_num
                # calculate fake negative, true positive, fake negative, and true negative rate
                fn = ((predict_labels == 0) & (labels != 0)).sum().item()
                tp = ((predict_labels != 0) & (labels != 0)).sum().item()
                tn = ((predict_labels == 0) & (labels == 0)).sum().item()
                fp = ((predict_labels != 0) & (labels == 0)).sum().item()
                recall = 0
                precision = 0
                if tp != 0:
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp)
                F1_score = 0
                if precision != 0 or recall != 0:
                    F1_score = 2 * recall * precision / (recall + precision)
                
                # back-propagate
                optim.zero_grad()
                train_loss.backward(retain_graph=True)
                # print(model.GCN1.layers[0].attn_n.grad)
                optim.step()

                target_list = []
                label_hats = None
                graph.ndata['h'] = th.zeros((graph.number_of_nodes(),options.out_dim),dtype=th.float).to(device)
                graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(device)
                graph.edges['cell'].data['e'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(device)
                feat_map = cnn(cnn_inputs.to(device)).reshape((1,-1)) if cnn is not None else None
                #if feat_map is not None: feat_map = feat_map.reshape((1,feat_map[0]*feat_map[1]))
                end = time()
                print("e{},{},b{}/{}, l:{:.3f}, r2:{:.3f}, r:{:.3f}, F1:{:.3f}".format(epoch,design,bidx,num_batch, train_loss.item(),train_r2.item(),recall,F1_score))
                flag = bidx % 50 == 0
                if flag or bidx==num_batch-1:
                    val_res,val_F1_score,val_r2 = validate(data_save_path,val_designs, device, model,cnn,beta,options)
                    if options.task == 'cls':
                        judgement = val_F1_score > max_F1_score 
                    elif options.task == 'reg':
                        judgement = val_r2 > max_r2
                    else:
                        assert False
                    #judgement = True
                    if judgement:
                        stop_score = 0
                        max_F1_score = val_F1_score
                        max_r2 = val_r2
                        print("Saving model.... ", os.path.join(options.model_saving_dir))
                        if os.path.exists(options.model_saving_dir) is False:
                            os.makedirs(options.model_saving_dir)
                        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                            parameters = options
                            pickle.dump((parameters, model,cnn), f)
                        print("Model successfully saved")
        





if __name__ == "__main__":
    
    seed = random.randint(1,10000)
    seed = 9294
    # th.set_deterministic(True)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    options = get_options()
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir,exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        train(options,seed)

