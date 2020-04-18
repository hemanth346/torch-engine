def get_misclassified(self, number=20):
    # predict_generator
    '''
        Generates output predictions for the input samples.
        predict(x, batch_size=None, verbose=0, steps=None, callbacks=None,
        max_queue_size=10, workers=1, use_multiprocessing=False)
    '''
    if self.data_loader:
        data_loader = self.data_loader

    misclassified_data = []
    misclassified_ground_truth = []
    misclassified_predicted = []
    self.model.eval()

    with torch.no_grad():
        # TODO : None object
        for data, target in data_loader:
            # move to respective device
            data, target = data.to(self.device), target.to(self.device)
            # inference
            output = self.model(data)

            # get predicted output and the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # get misclassified list for this batch
            misclassified_list = (target.eq(pred.view_as(target)) == False)
            misclassified = data[misclassified_list]
            ground_truth = pred[misclassified_list]
            predicted = target[misclassified_list]

            # stitching together
            misclassified_data.append(misclassified)
            misclassified_ground_truth.append(ground_truth)
            misclassified_predicted.append(predicted)

            # stop after enough false positives
            if len(misclassified_data) >= number:
                break

    # converting to torch tensors
    misclassified_data = torch.cat(misclassified_data)
    misclassified_ground_truth = torch.cat(misclassified_ground_truth)
    misclassified_predicted = torch.cat(misclassified_predicted)

    self.misclassified_data = misclassified_data
    self.misclassified_ground_truth = misclassified_ground_truth
    self.misclassified_predicted = misclassified_predicted

    return misclassified_data, misclassified_ground_truth, misclassified_predicted



def plot_misclassifications(self, target_layers):

    misclassified = []
    misclassified_target = []
    misclassified_pred = []

    model, device = self.trainer.model, self.trainer.device

    # set the model to evaluation mode
    model.eval()

    # turn off gradients
    with torch.no_grad():
        for data, target in self.trainer.test_loader:
            # move them to respective device
            data, target = data.to(device), target.to(device)

            # do inferencing
            output = model(data)

            # get the predicted output
            pred = output.argmax(dim=1, keepdim=True)

            # get the current misclassified in this batch
            list_misclassified = (target.eq(pred.view_as(target)) == False)
            batch_misclassified = data[list_misclassified]
            batch_mis_pred = pred[list_misclassified]
            batch_mis_target = target[list_misclassified]

            # batch_misclassified =

            misclassified.append(batch_misclassified)
            misclassified_pred.append(batch_mis_pred)
            misclassified_target.append(batch_mis_target)

    # group all the batched together
    misclassified = torch.cat(misclassified)
    misclassified_pred = torch.cat(misclassified_pred)
    misclassified_target = torch.cat(misclassified_target)

    logger.info('Taking {25} samples')
    # get 5 images
    data = misclassified[:25]
    target = misclassified_target[:25]
