import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help = 'the learning rate for training. Type: float.',default=1e-3)
    parser.add_argument("--batch_size", type=int, help = 'the number of samples in each training batch. Type: int',default=1350)
    parser.add_argument("--num_epoch", type=int, help='Type: int; number of epoches that the training procedure runs. Type: int',default=1000)
    parser.add_argument("--in_dim", type=int, help='the dimension of the input feature. Type: int',default=512)
    parser.add_argument("--out_dim", type=int, help='the dimension of the output embedding. Type: int',default=128)
    parser.add_argument("--cell_feat_dim", type=int, help='the dimension of the cell feature. Type: int', default=42)
    parser.add_argument("--net_feat_dim", type=int, help='the dimension of the net feature. Type: int', default=3)
    parser.add_argument("--hidden_dim", type=int, help='the dimension of the intermediate GNN layers. Type: int',default=256)

    parser.add_argument('--cnn_input_dim',type=int,default=512)
    parser.add_argument('--cnn_outdim',type=int,default=128)
    parser.add_argument('--map_size',type=int,default=128)
    parser.add_argument("--gcn_dropout", type=float,help='dropout rate for GNN layers. Type: float', default=0)
    parser.add_argument("--mlp_dropout", type=float, help='dropout rate for mlp. Type: float',default=0)
    parser.add_argument("--weight_decay", type=float, help='weight decay. Type: float',default=0)
    parser.add_argument("--model_saving_dir", type=str, help='the directory to save the trained model. Type: str',default='../models/asap7-designs')
    parser.add_argument("--preprocess",help='decide whether to run the preprocess procedure or not. If set True, then a preprocess procedure'
                                            ' (generating dataset + initialize model)will be carried out; '
                                            'Else a normal training procedure will be carried out.'
                                            'Type: sote_true',action='store_true')
    parser.add_argument("--n_fcn",type=int,help='the number of full connected layers of the mlp. Type: int',default=3)
    parser.add_argument("--alpha",type=float,help='the weight of the cost-sensitive learning. Type: float',default=1.0)
    parser.add_argument("--change_lr",help='Decide to change to learning rate. Type: float',action='store_true')
    parser.add_argument("--change_alpha",help='Decide to change alpha. Type: float',action='store_true')
    parser.add_argument("--gpu",type=int,help='index of gpu. Type: int',default=0)
    #parser.add_argument('--balanced',action='store_true',help = 'decide whether to balance the training dataset (using oversampling) or not; Type: store_true')
    parser.add_argument('--nlabels',type=int,help='number of prediction classes. Type: int',default=1)
    parser.add_argument('--os_rate',help='the oversampling rate. Type: int',type=int,default=1)
    parser.add_argument('--beta',type=float,default=0.5,help='choose the threshold for binary classification to make a trade-off between recall and precision. Type: float')
    parser.add_argument('--data_save_path',type=str,help='the directory that contains the dataset. Type: str',default='../datasets/asap7-designs')
    parser.add_argument('--rawdata_path',type=str,default='../rawdata/example')
    #parser.add_argument('--data_info_txt',type=str, help='the file that saves the information of the data to parse')
   #parser.add_argument('--data_usage',type=str,help='decide whether to generate train dataset or test dataset')
    parser.add_argument('--predict_path',type=str,help='the directory used to save the prediction result. Type: str',default='../prediction/example')
    parser.add_argument('--droplast',action='store_true')
    parser.add_argument('--feat_reduce',type=int,nargs='+',default=[6,1])
    parser.add_argument('--no_cnn',action='store_true')
    parser.add_argument('--no_gnn',action='store_true')
    parser.add_argument('--masking',type=str,default='critical')
    parser.add_argument('--design',type=str)
    parser.add_argument('--unet',action='store_true',help='decide whether the layoutnet use the unet architecture')
    parser.add_argument('--pooling',type=str,default='max',help='the pooling type for layoutnet')
    parser.add_argument('--norm',action='store_true',help='decide whether to normalize the input feature or not')
    parser.add_argument('--task',type=str,default='reg',help="decice a classification task or regression task, valid values: ['cls','reg']")
    parser.add_argument('--attn',action='store_true',help='decide whether to apply attention mechanism in GNN')
    parser.add_argument('--num_heads',type=int,default=1,help='Decide the number of heads for attention mechanism')
    options = parser.parse_args(args)

    
    return options

