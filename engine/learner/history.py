from engine.utils.plots import show_lr_metrics, show_metrics


def plot_history(model, metrics=['train_acc', 'train_loss', 'val_acc', 'val_loss'], show_lr=False, nplots=2):
    data = []
    for metric in metrics:
        data.append(model.metrics[metric])
    if show_lr:
        show_lr_metrics(data, metrics, lr_history=model.metrics['lr'], nplots=nplots)
    else:
        show_metrics(data, metrics, nplots)

