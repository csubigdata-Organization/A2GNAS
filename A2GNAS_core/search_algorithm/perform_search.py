import torch
import numpy as np
from A2GNAS_core.model.logger import gnn_architecture_performance_save
from A2GNAS_core.model.asymm_model import AsymmModel
from A2GNAS_core.model.fullsupervised_test import scratch_train_each_epoch, eval_dataset


def estimation(gnn_architecture_list, args, graph_data):
    performance = []

    for gnn_architecture in gnn_architecture_list:
        res = search_train(graph_data=graph_data,
                           gnn_architecture=gnn_architecture,
                           args=args)
        performance.append(res)

        gnn_architecture_performance_save(gnn_architecture, res, args.data_save_name)

    return performance


def search_train(graph_data, gnn_architecture, args):
    valid_accs = []

    asymm_model = AsymmModel(gnn_architecture, args).to(args.device)
    asymm_optimizer = torch.optim.Adam([{'params': asymm_model.parameters()}],
                                       lr=args.learning_rate,
                                       weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(asymm_optimizer, float(args.train_epoch),
                                                           eta_min=args.learning_rate_min)


    for epoch in range(args.train_epoch):
        train_loader = graph_data[-3]
        scratch_train_each_epoch(asymm_model, asymm_optimizer, train_loader, args)
        scheduler.step()

        val_loader = graph_data[-2]
        valid_acc = eval_dataset(asymm_model, val_loader, args)
        valid_accs.append(valid_acc)


    return_valid_acc = np.mean(valid_accs)

    return return_valid_acc
