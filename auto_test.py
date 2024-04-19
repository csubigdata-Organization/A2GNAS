import os
import numpy as np
import torch
import random
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from dataset import load_data
from A2GNAS_core.model.fullsupervised_test import fullsupervised_scratch_train

graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K']

def arg_parse():

    parser = argparse.ArgumentParser("A2GNAS.")
    parser.add_argument('--data_name', type=str, default='MUTAG', help='location of the data corpus')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs_scratch', type=int, default=100, help='num of test epochs from scratch')
    parser.add_argument('--hidden_dim', type=int, default=64, help='default hidden_dim for gnn model')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--initial_num', type=int, default=100)
    parser.add_argument('--sharing_num', type=int, default=10)
    parser.add_argument('--search_epoch', type=int, default=8)
    parser.add_argument('--train_epoch', type=int, default=100, help='the number of train epoch for sampled model')
    parser.add_argument('--return_top_k', type=int, default=5, help='the number of top model for testing')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = arg_parse()
set_seed(args.seed)

device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
args.device = device

print(args.data_name, 'dataset')

def get_test_architecture(data_name):
    if data_name == "MUTAG":
        target_architecture = ['autogcl', 'GIN_PM', 'GCPool', 'global_add', '7', '5']
    elif data_name == "COX2":
        target_architecture = ['attribute_masking', 'Graph_PM', 'None', 'global_max', '4', '5']
    elif data_name == "NCI109":
        target_architecture = ['attribute_masking', 'Graph_PM', 'None', 'global_max', '7', '3']
    elif data_name == "DD":
        target_architecture = ['edge_perturbation', 'GAT_PM', 'SAGPool', 'global_add', '2', '4']
    elif data_name == "PROTEINS":
        target_architecture = ['node_dropping', 'SAGE_PM', 'TopKPool', 'global_add', '3', '4']
    else:
        raise Exception("Wrong dataset name")
    return target_architecture


args.graph_classification_dataset = graph_classification_dataset
if args.data_name in args.graph_classification_dataset:
    graph_data, num_nodes = load_data(args.data_name, batch_size=args.batch_size, split_seed=args.seed)
    num_features = graph_data[0].num_features
    if args.data_name == 'COLORS-3':
        num_classes = 11
    else:
        num_classes = graph_data[0].num_classes
    args.num_features = num_features
    args.num_classes = num_classes
    args.learning_type = 'fullsupervised'
    args.data_save_name = args.data_name + '_' + args.learning_type

    target_architecture = get_test_architecture(args.data_name)
    print(35 * "=" + " the testing start " + 35 * "=")
    print("dataset:", args.data_name)
    print("test gnn architecture", target_architecture)

    ## train from scratch
    test_repeat = 5
    for i in range(test_repeat):
        valid_acc, test_acc, test_acc_std = fullsupervised_scratch_train(target_architecture, graph_data, args=args)
        print("{}-th run in all {} runs || Test_acc:{}±{}".format(i + 1, test_repeat, test_acc, test_acc_std))
    print(35 * "=" + " the testing ending " + 35 * "=")
