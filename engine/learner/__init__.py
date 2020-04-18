import torch
from tqdm import tqdm

# properties
cuda = torch.cuda.is_available()

# # inputs
# network = None
# train_loader = None
# test_loader = None
# criterion = None
# optimizer = None
# scheduler = None
# epochs = 20

# import os
# from .data_loader import CIFAR10Data
# from .trainer import Trainer
# from .plots import plot_history
# from .visualize import ShowData, Validator


def make_train_step(model, loss_fn, optimizer, device):
    # Builds function that performs a step in the train loop
    def train_step(train_loader):
        epoch_train_losses = []
        epoch_train_acc = []
        correct, processed = 0, 0

        pbar = tqdm(train_loader, dynamic_ncols=True)

        for batch_id, (x, y) in enumerate(pbar):
            if batch_id == 3:
                break
            # send mini-batches to the device
            # where the model "lives"
            x = x.to(device)
            y = y.to(device)

            # Sets model to TRAIN mode
            model.train()

            # zeroes gradients for optimizer params
            # i.e set grad attribute to zero
            optimizer.zero_grad()

            # Makes predictions
            y_pred = model(x)
            # Computes loss
            loss = loss_fn(y_pred, y)
            # Computes gradients
            loss.backward()

            # Updates parameters
            optimizer.step()

            # performs the updates
            batch_loss = loss.item()
            # total_train_loss += batch_loss

            # Inference
            _, predicted = y_pred.max(1)

            # Total stat until current batch
            processed += len(x)
            correct += predicted.eq(y).sum().item()

            # Total accuracy until current batch
            train_accuracy = correct/processed

            progress_desc = (f'batch_id : {batch_id} | Batch loss : {batch_loss :.5f} '
                  f'accuracy : {100.*train_accuracy :.2f} ({correct}/{processed})')
            pbar.set_description(progress_desc)
            # logger.info(progress_desc)

            # append batch stats
            epoch_train_losses.append(batch_loss)
            epoch_train_acc.append(100. * train_accuracy)
        # model.update_train_metrics(epoch_train_losses, epoch_train_acc)
        return epoch_train_losses, epoch_train_acc
    # Returns the function that will be called inside the train loop
    return train_step


def make_test_step(model, loss_fn, scheduler, device):
    def test_step(test_loader):
        # print('Call : ', __name__)
        epoch_val_losses = []
        epoch_val_acc = []
        correct, processed = 0, 0
        epoch_loss = 0
        with torch.no_grad():
            for batch_id, (x_val, y_val) in enumerate(test_loader):
                if batch_id == 3:
                    break
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                model.eval()

                y_pred = model(x_val)
                val_loss = loss_fn(y_pred, y_val)

                # extract value as float
                batch_loss = val_loss.item()
                _, predicted = y_pred.max(1)

                # same as +=len(x_val)
                processed += y_val.size(0)
                correct += predicted.eq(y_val).sum().item()

                # epoch avg_accuracy
                val_acc = correct/processed
                epoch_loss += batch_loss
                avg_loss = epoch_loss/len(test_loader)
                epoch_val_losses.append(avg_loss)
                epoch_val_acc.append(100.*val_acc)

        print(f'Loss : {avg_loss :.6f} | Accuracy : {100.*val_acc :.2f}')

        # https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        elif scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler.step(round(avg_loss, 6))

        return epoch_val_losses, epoch_val_acc

    return test_step


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


def classwise_accuracy(model, test_loader, classes, device='cpu'):
    '''
        Class wise total accuracy
    :param classes:
    :return:
    '''
    class_total = list(0. for i in range(10))
    class_correct = list(0. for i in range(10))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:<10} : {(100 * class_correct[i] / class_total[i]):.2f}%')



def get_misclassified(model, data_loader, number=20, device='cpu'):
    # predict_generator
    '''
        Generates output predictions for the input samples.
        predict(x, batch_size=None, verbose=0, steps=None, callbacks=None,
        max_queue_size=10, workers=1, use_multiprocessing=False)
    '''

    misclassified_data = []
    misclassified_ground_truth = []
    misclassified_predicted = []
    model.eval()

    count = 0
    with torch.no_grad():
        for data, target in data_loader:
            # move to respective device
            data, target = data.to(device), target.to(device)
            # inference
            output = model(data)

            # get predicted output and the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # get misclassified list for this batch
            misclassified_list = (target.eq(pred.view_as(target)) == False)
            misclassified = data[misclassified_list]
            ground_truth = pred[misclassified_list]
            predicted = target[misclassified_list]
            count += misclassified.shape[0]
            # stitching together
            misclassified_data.append(misclassified)
            misclassified_ground_truth.append(ground_truth)
            misclassified_predicted.append(predicted)
            # stop after enough false positives
            if count >= number:
                break

    # converting to torch
    # clipping till given number if more count from batch
    misclassified_data = torch.cat(misclassified_data)[:number]
    misclassified_ground_truth = torch.cat(misclassified_ground_truth)[:number]
    misclassified_predicted = torch.cat(misclassified_predicted)[:number]

    # print('shape (misclassified_data), shape (misclassified_ground_truth), shape (misclassified_predicted) : ', misclassified_data.shape, (misclassified_ground_truth.shape), (misclassified_predicted.shape))
    return misclassified_data, misclassified_ground_truth, misclassified_predicted