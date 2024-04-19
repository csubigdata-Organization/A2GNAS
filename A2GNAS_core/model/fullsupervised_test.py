import torch
import torch.nn.functional as F
from dataset import load_k_fold
from A2GNAS_core.model.asymm_model import AsymmModel


def fullsupervised_scratch_train(gnn_architecture, graph_data, args):
    valid_accs = []
    test_accs = []


    folds = 20
    k_folds_data = load_k_fold(graph_data[0], folds, args.batch_size)
    argmax_list = []
    for fold, fold_data in enumerate(k_folds_data):

        asymm_model = AsymmModel(gnn_architecture, args).to(args.device)

        asymm_optimizer = torch.optim.Adam([{'params': asymm_model.parameters()}],
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(asymm_optimizer, float(args.epochs_scratch),
                                                               eta_min=args.learning_rate_min)

        print('###fold {}, train/val/test:{},{},{}'.format(fold+1, len(fold_data[-3].dataset),len(fold_data[-2].dataset),len(fold_data[-1].dataset)))
        max_acc = 0
        max_index = 0
        for epoch in range(1, args.epochs_scratch + 1):
            train_loader = fold_data[-3]
            scratch_train_each_epoch(asymm_model, asymm_optimizer, train_loader, args)
            scheduler.step()

            val_loader = fold_data[-2]
            valid_acc = eval_dataset(asymm_model, val_loader, args)
            valid_accs.append(valid_acc)

            test_loader = fold_data[-1]
            test_acc = eval_dataset(asymm_model, test_loader, args)
            test_accs.append(test_acc)

            if valid_acc >= max_acc:
                max_acc = valid_acc
                max_index = epoch-1

            if epoch % 10 == 0:
                print('fold:{}, epoch:{}, valid_acc:{:.4f}, test_acc:{:.4f}'.format(fold+1,epoch,valid_acc,test_acc))
        argmax_list.append(max_index)

    valid_accs = torch.tensor(valid_accs).view(folds, args.epochs_scratch)
    test_accs = torch.tensor(test_accs).view(folds, args.epochs_scratch)

    # max_valid_acc
    valid_accs_argmax = valid_accs[torch.arange(folds, dtype=torch.long), argmax_list] * 100
    valid_acc_mean = round(valid_accs_argmax.mean().item(), 2)
    test_accs_argmax = test_accs[torch.arange(folds, dtype=torch.long), argmax_list] * 100
    test_acc_mean = round(test_accs_argmax.mean().item(), 2)
    test_acc_std = round(test_accs_argmax.std().item(), 2)
    print('test_accs:', test_accs_argmax)

    return valid_acc_mean, test_acc_mean, test_acc_std

def scratch_train_each_epoch(asymm_model, asymm_optimizer, train_loader, args):
    asymm_model.train()

    for train_data in train_loader:
        train_data = train_data.to(args.device)

        output_augment = asymm_model(train_data, augment=True)
        output_origin = asymm_model(train_data, augment=False)

        cls_loss_augment = F.nll_loss(output_augment, train_data.y.view(-1))
        cls_loss_origin = F.nll_loss(output_origin, train_data.y.view(-1))
        cls_loss = cls_loss_augment + cls_loss_origin

        asymm_optimizer.zero_grad()
        cls_loss.backward()
        asymm_optimizer.step()


def eval_dataset(asymm_model, data_loader, args):
    asymm_model.eval()
    performance_one_epoch = 0

    for eval_data in data_loader:
        eval_data = eval_data.to(args.device)
        with torch.no_grad():
            output = asymm_model(eval_data, augment=False)
        performance_one_epoch += output.max(1)[1].eq(eval_data.y.view(-1)).sum().item()
    performance_one_epoch = performance_one_epoch / len(data_loader.dataset)

    return performance_one_epoch
