import torch

from .step import make_train_step, make_test_step

def fit(model, epochs, train_loader, test_loader, loss_fn, optimizer, scheduler, device):
    train_step = make_train_step(model, loss_fn, optimizer, device)
    test_step = make_test_step(model, loss_fn, scheduler, device)
    for epoch in range(1, epochs+1):
        print(f'Epoch : {epoch}', sep=', ')
        for group in optimizer.param_groups:
            # from lr_scheduler source codes
            # https://github.com/pytorch/pytorch/issues/2829#issuecomment-331800609
            print(f'LR : {group["lr"]}')
            model.metrics['lr'].append(group["lr"])

        train_loss, train_acc = train_step(train_loader)
        val_loss, val_acc = test_step(test_loader)

        # update model metrics dict
        model.update_train_metrics(train_loss, train_acc)
        model.update_test_metrics(val_loss, val_acc)


