from tqdm.auto import tqdm
import torch


def make_train_step(model, loss_fn, optimizer, device, progress=True):
    # Builds function that performs a step in the train loop
    def train_step(train_loader):
        epoch_train_losses = []
        epoch_train_acc = []
        correct, processed = 0, 0
        if progress:
            pbar = tqdm(train_loader, dynamic_ncols=True)
        else:
            pbar = train_loader
        for batch_id, (x, y) in enumerate(pbar):
            # send mini-batches to the device
            # where the model "lives"
            x = x.to(device)
            y = y.to(device)

            # Sets model to TRAIN mod   e
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
