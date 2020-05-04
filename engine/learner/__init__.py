import os
import torch
from engine.utils.plots import plot_history, show_misclassified_images
from engine.learner.step import make_train_step, make_test_step


class Learner(object):
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch = 0

    def fit(self, epochs):
        train_step = make_train_step(self.model, self.loss_fn, self.optimizer, self.device)
        test_step = make_test_step(self.model, self.loss_fn, self.scheduler, self.device)
        for epoch in range(1, epochs+1):
            print(f'Epoch : {epoch}', sep=', ')
            for group in self.optimizer.param_groups:
                # from lr_scheduler source codes
                # https://github.com/pytorch/pytorch/issues/2829#issuecomment-331800609
                print(f'LR : {group["lr"]}')
                self.model.metrics['lr'].append(group["lr"])

            train_loss, train_acc = train_step(self.train_loader)
            val_loss, val_acc = test_step(self.test_loader)

            # update model metrics dict
            self.epoch = epoch
            self.model.update_train_metrics(train_loss, train_acc)
            self.model.update_test_metrics(val_loss, val_acc)

    def plot_history(self, metrics=['train_acc', 'train_loss', 'val_acc', 'val_loss'],
                     show_lr=False, nplots=2):
        # TODO : Add plot fig size
        plot_history(self.model, metrics=metrics, show_lr=show_lr, nplots=nplots)

    def show_misclassified(self, classes, mean, std, number=20,
                           device='cpu', ncols=5, figsize=(10, 6)):
        show_misclassified_images(self.model, self.test_loader, classes=classes, mean=mean, std=std,
                                  number=number, device=device, ncols=ncols, figsize=figsize)

    def save_checkpoint(self, name, path='.'):
        if os.path.splitext(name)[1] in ['.pt', '.pth']:
            state_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss_fn': self.loss_fn,
                'epoch': self.epoch,
                'metrics': self.model.metrics
            }
            path = os.path.join(os.path.abspath(path), name)
            torch.save(state_dict, path)
            print('Checkpoint saved to ', path)
            return True
        else:
            print('Checkpoint file should end with either ".pt" or ".pth" \n Checkpoint nor created...')
            return False


def load_checkpoint(learner:Learner, path):
    path = os.path.abspath(path)
    if os.path.splitext(path)[1] not in ['.pt', '.pth']:
        print('Checkpoint file should end with either ".pt" or ".pth" \n Checkpoint nor created...')
        return learner
    if os.path.isfile(path):
        print("loading checkpoint from ", path)
        checkpoint = torch.load(path)
        learner.model.load_state_dict(checkpoint['model_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        learner.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        learner.loss_fn = checkpoint['loss_fn']
        learner.epoch = checkpoint['epoch']
        learner.model.metrics = checkpoint['metrics']
        print("Done.. Loaded checkpoint '{0}' (epoch {1})".format(path, checkpoint['epoch']))
    else:
        print('No checkpoint found at ', path)
    return learner
