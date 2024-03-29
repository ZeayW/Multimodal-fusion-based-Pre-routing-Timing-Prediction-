import random
from importlib.resources import path
from lib2to3.pytree import Node
from tracemalloc import start
from dataset import *
from options import get_options
from model import *
from TimeConv import *
import dgl
import pickle
import numpy as np
import os
from time import time
from random import shuffle
import itertools
from MyDataloader import *
from torchmetrics import R2Score
from train import norm, judge_critical

device = th.device("cuda:"+str(get_options().gpu) if th.cuda.is_available() else "cpu")
R2_score = R2Score().to(device)

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
        assert False, 'no model!'       
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
    if cnn is not None:cnn = cnn.to(device)
    # model = nn.DataParallel(model, device_ids=[0,1])
    # cnn = nn.DataParallel(cnn, device_ids=[0,1])
    
    print('Model and hyper-parameters successfully loaded!')
    return param, model, cnn


def load_data(data_path,usage,init_feat_dim,os_rate,feat_reduce,if_norm):

    """

    load the data

    :param dataset_file: str
            a pickle file that saves the dataset
    :param init_feat_dim: int
            the dimension of the initial feature
    :return:
        updated_dataset: List[(graph, topo_levels)]
            where graph is the DAG representation of a circuit,
            and topo_levels are the calculated topological levels
    """
    assert usage in ['train','test'], "Wrong data usage! Should be either 'train' or 'test'."
    design_list_file = os.path.join(data_path,'{}data_list.txt'.format(usage))
    assert os.path.exists(design_list_file), \
        "Can not find the traindata list txt '{}'".format(design_list_file)

    with open(design_list_file,'r') as f:
        lines = f.readlines()
        design_list = [l.replace('\n','') for l in lines]
    
    print('--- {} designs: '.format(usage),design_list)
    datasets = []
    caseid2design = {}
    for i,design in enumerate(design_list):
        caseid2design[i] = design
        dataset_file = os.path.join(data_path,'{}.pkl'.format(design))
        graph,topo_levels,path_masks,path2level,path2endpoint,critical_paths,cnn_inputs = th.load(dataset_file)
        #with open(dataset_file,'rb') as f:
            #graph,topo_levels,path_masks,path2level,path2endpoint,critical_paths,cnn_inputs = pickle.load(f)
        graph.ndata['h'] = th.zeros((graph.number_of_nodes(), init_feat_dim), dtype=th.float)
        if feat_reduce is not None:
            if feat_reduce[1] != 0: 
                graph.ndata['net_feat'] = graph.ndata['net_feat'][:,:-feat_reduce[1]]
            if feat_reduce[0]!=0:
                graph.ndata['cell_feat'] = graph.ndata['cell_feat'][:,:-feat_reduce[0]]
            #print(graph.ndata['cell_feat'][:5])
        # normalize all the features, so that the value of different feature will not differ greatly
        if if_norm:
            graph.ndata['cell_feat'] = norm(graph.ndata['cell_feat'],num_ctypes)
            graph.ndata['net_feat'] = norm(graph.ndata['net_feat'],num_ctypes)
      
        if type(cnn_inputs)== np.ndarray:
            cnn_inputs = th.from_numpy(cnn_inputs).float()
        #cnn_inputs = th.unsqueeze(cnn_inputs,dim=0)
        paths = list(range(len(graph.ndata['end'][graph.ndata['end'].squeeze()==1])))
        #print(len(paths),paths[:10])
        #if os_rate!=0:
        #    for _ in range(os_rate):
        #        paths.extend(critical_paths)
        split_file = os.path.join(data_path,'{}_split.pkl'.format(design))
        #with open(split_file,'rb') as f:
        #    val_paths,test_paths = pickle.load(f)
        #paths = test_paths
        path_dataset = PathDataset(paths)

        datasets.append(
            (path_dataset,graph,path2level,path2endpoint,topo_levels,cnn_inputs,path_masks)
        )
    
    return datasets,caseid2design

def test(model_save_path,loader,device,model,cnn,Loss,beta,options,case2design):
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
    predict_save_dir = os.path.join(options.model_saving_dir,'predict_critical')
    res_save_path = os.path.join(model_save_path,'predict.txt')

    overall_loss, overall_acc, overall_recall, overall_precision, overall_f1,  overall_r2 = 0, 0.0, 0, 0, 0, 0
    # runtime = 0
    res = []
    with th.no_grad():
        # load validation data, one batch at a time
        # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
        # and output neighborhoods (out_block).
        # The dst_nodes of the last block of in_block/out_block is the central nodes.
        case_idx = 0
        for path_dataset,graph,path2level,path2endpoint,topo_levels,cnn_inputs,path_masks in loader:
            
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
                path_ids = list(path_ids.numpy().tolist())
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
                    targets = sampled_ends.get(level_id,[])
                    paths = sampled_paths.get(level_id,[])
                    target_list.extend(targets)
                    #target_paths.extend()
                    count_target += len(target_list)

                    if options.no_cnn or len(paths) == 0:
                        path_map = None
                    else:
                        path_mask = th.index_select(path_masks,0,th.tensor(paths)).to(device)
                        path_map = path_mask.to_dense()*feat_map
                        #path_map = path_map.view(-1,path_map.shape[1]*path_map.shape[2])
                
                    cur_label_hats = model(graph,nodes,eids,targets,level_id,path_map)

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
            # positive_pids = th.tensor(target_paths)[predict_labels==1].numpy().tolist()
            # design = case2design[case_idx]
            # with open(os.path.join(predict_save_dir,'{}.pkl'.format(design)),'wb') as f:
            #     pickle.dump(positive_pids,f)

            #print(cnn.down3.maxpool_conv.weights)
            end = time()
            runtime = end-start_time
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
            
            # overall_correct += correct
            # overall_fn  += fn
            # overall_fp += fp
            # overall_tn += tn
            # overall_tp += tp
            # overall_num += total_num
            print('case {}, runtime: {}'.format(case_idx,runtime))
            case_idx += 1
            print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
            print("\tloss:{:.3f}, r2:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(test_loss, test_r2,acc,recall, F1_score))
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

    print('overall val')
    #print("\ttp:", overall_tp, " fp:", overall_fp, " fn:", overall_fn, " tn:", overall_tn, " precision:", round(overall_precision, 3))
    print("\tloss:{:.3f}, r2:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(overall_loss, overall_r2, overall_acc,overall_recall, overall_f1))
    
    with open(res_save_path,'a') as f:
        f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n"
                .format(overall_loss, overall_r2,overall_acc,overall_recall,overall_precision,overall_f1))
    return res,overall_f1,overall_r2


def test_old(model_save_path,loader,device,model,cnn,Loss,beta,options,case2design):
    r"""

    test the model

    :param loader:
        the data loader to load the test dataset
    :param label_name:
        target label name
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
    predict_save_dir = os.path.join(options.model_saving_dir,'predict_critical')
    res_save_path = os.path.join(model_save_path,'predict.txt')

    overall_loss, overall_acc, overall_recall, overall_precision, overall_f1,  overall_r2 = 0, 0.0, 0, 0, 0, 0
    # runtime = 0
    with open(res_save_path,'w') as f:
        pass
    with th.no_grad():
        # load validation data, one batch at a time
        # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
        # and output neighborhoods (out_block).
        # The dst_nodes of the last block of in_block/out_block is the central nodes.
        case_idx = 0
        for path_dataset,graph,path2level,path2endpoint,topo_levels,cnn_inputs,path_masks in loader:
            total_num, total_loss, correct, fn, fp, tn, tp, total_r2 = 0, 0.0, 0, 0, 0, 0, 0,0
            
            # optim.zero_grad()
            start_time = time()
            feat_map = cnn(cnn_inputs.to(device)).squeeze(0).squeeze(0) if cnn is not None else None
            # transfer the data to GPU
            graph = graph.to(device)
            count_target = 0
            label_hats = None
            target_list = []
            target_paths = []
            # first_level_nodes = topo_levels[0][0]
            # init_message = graph.ndata['h'][first_level_nodes]
            count = 0
            #endpoint2pid = {}
            for level_id, (nodes,targets,path_ids) in enumerate(topo_levels[1:]):
                target_list.extend(targets)
                paths = sampled_paths.get(level_id,[])
                target_paths.extend(path_ids)
                count_target += len(target_list)
                
                if options.no_cnn or len(paths) == 0:
                        path_map = None
                else:
                    path_mask = th.index_select(path_masks,0,th.tensor(paths)).to(device)
                    path_map = path_mask.to_dense()*feat_map
                cur_label_hats = model(graph,nodes,targets,level_id,path_map)

                if label_hats is None:
                    label_hats = cur_label_hats
                else:
                    label_hats = th.cat((label_hats,cur_label_hats),dim=0)

            # the ground-truth labels
            labels = graph.ndata['label'][target_list].squeeze()
            # predicted labels
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
            
            positive_pids = th.tensor(target_paths)[predict_labels==1].numpy().tolist()
            design = case2design[case_idx]
            with open(os.path.join(predict_save_dir,'{}.pkl'.format(design)),'wb') as f:
                pickle.dump(positive_pids,f)


            # calculated loss:
            end = time()
            runtime = end-start_time
           
            #test_loss = Loss(label_hats, labels)

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
            verall_recall += recall
            overall_f1 += F1_score
            overall_acc += acc
            overall_precision += precision
            # overall_correct += correct
            # overall_fn  += fn
            # overall_fp += fp
            # overall_tn += tn
            # overall_tp += tp
            # overall_num += total_num
            
            print('case {}, runtime: {}'.format(case_idx,runtime))
            case_idx += 1
            print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
            print("\tloss:{:.3f}, r2:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}"
                        .format(test_loss,test_r2, acc,recall, F1_score))
            with open(res_save_path,'a') as f:
                f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n"
                        .format(test_loss, test_r2, acc,recall, precision,F1_score))
    # calculate the overall loss / accuracy
    num_case = case_idx -1
    overall_loss = overall_loss / num_case
    overall_acc = overall_acc / num_case
    overall_r2 = overall_r2 / num_case
    overall_f1 = overall_f1 / num_case
    overall_recall = overall_recall / num_case
    overall_precision = overall_precision / num_case
    print('overall test')
    #print("\ttp:", overall_tp, " fp:", overall_fp, " fn:", overall_fn, " tn:", overall_tn, " precision:", round(overall_precision, 3))
    print("\tloss:{:.3f}, r2:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(overall_loss, overall_r2,overall_acc,overall_recall, overall_f1))

    with open(res_save_path,'a') as f:
        f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n"
                .format(overall_loss, overall_r2,overall_acc,overall_recall,overall_precision,overall_F1_score))
    return [overall_loss, overall_r2,overall_acc,overall_recall,overall_precision,overall_F1_score]

def main(options):
    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:"+str(options.gpu) if th.cuda.is_available() else "cpu")

    # you can define your dataset file here
    data_save_path = options.data_save_path
    print(data_save_path)
    os.makedirs(os.path.join(options.model_saving_dir,'predict_critical'),exist_ok=True)
    # load the model
    options, model,cnn = load_model(device, options)
    print('Hyperparameters are listed as follows:')
    print(options)
    print('The model architecture is shown as follow:')
    print(model)
    print(cnn)
    print("----------------Loading data----------------")
    test_data_file = os.path.join(data_save_path, 'test.pkl')
    test_dataset,case2design = load_data(data_save_path,'train',options.out_dim,options.os_rate,options.feat_reduce,options.norm)
    Loss = nn.CrossEntropyLoss() if options.task == 'cls' else nn.MSELoss() 
    beta = options.beta
    
    test(options.model_saving_dir,test_dataset,device,model,cnn,Loss,beta,options,case2design)

if __name__ == "__main__":
    
    main(get_options())
