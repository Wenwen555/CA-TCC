import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from models.loss import NTXentLoss, SupConLoss

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        return loss

def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # for fine_tuned with labels
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss_mape, valid_loss_mse, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss_mape)
            # scheduler.step(valid_loss_mse)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train MSE Loss     : {train_loss:2.6f}\t | \tTrain Accuracy     : {train_acc:2.6f}\n'
                     f'Valid MSE Loss     : {valid_loss_mse:2.6f}\t | \tValid Accuracy     : {valid_loss_mse:2.6f}\n'
                     f'Valid MAPE Loss    : {valid_loss_mape:2.6f}\t | \tValid Accuracy     : {valid_loss_mse:2.6f}\n'
                     f'lr                 : {scheduler.get_last_lr()}\n'
                     )


    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test MAPE loss      :{test_loss:2.6f}\t | Test MSE loss      : {test_acc:2.6f}')

    logger.debug("\n################## Training is Done! #########################")


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))

def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    # encode_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.float().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # data = encode_model(data)
        # aug1 = encode_model(aug1)
        # aug2 = encode_model(aug2)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()
        # encode_model_optimizer.zero_grad()

        if training_mode == "self_supervised" or training_mode == "SupCon" or training_mode == "supervised_with_contrast":
            # encode_version of raw data.
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            # count_parameters(model)
            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2

        elif training_mode == "supervised_with_contrast":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss_con = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2
            predictions, features = model(data)
            loss_supervised = criterion(predictions, labels.view(-1, 1))
            loss = loss_supervised/loss_supervised.detach() + loss_con/(loss_con/loss_supervised).detach()

        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            Sup_contrastive_criterion = SupConLoss(device)

            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             labels) * lambda2

        else:
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels.view(-1, 1))
            # total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_acc.append(loss.item())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()
        # encode_model_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()
    # encode_model.eval()

    total_loss_mape = []
    total_loss_mse = []
    total_acc = []

    # criterion = nn.MSELoss()
    criterion_1 = MAPELoss()
    criterion_2 = nn.MSELoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.float().to(device)
            # data = encode_model(data)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss_mape = criterion_1(predictions, labels.view(-1, 1))
                loss_mse = criterion_2(predictions, labels.view(-1, 1))

                # total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_acc.append(loss_mape.item())
                total_loss_mape.append(loss_mape.item())
                total_loss_mse.append(loss_mse.item())

                # pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # outs = np.append(outs, pred.cpu().numpy())
                # trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss_mape = torch.tensor(total_loss_mape).mean()  # average loss
        total_loss_mse = torch.tensor(total_loss_mse).mean()
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss_mape, total_loss_mse, outs, trgs




def gen_pseudo_labels(model, dataloader, device, experiment_log_dir):
    from sklearn.metrics import accuracy_score
    model.eval()
    # softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []
    pseudo_dataset = []

    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).to(device)

            # forward pass
            predictions, features = model(data)
            # normalized_preds = softmax(predictions)
            # pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            pseudo_labels = predictions.squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()

    data_save["samples"] = all_data
    data_save["labels"] = torch.tensor(torch.from_numpy(all_pseudo_labels))
    # pseudo_dataset[bat]['summary']['QD'] = all_pseudo_labels
    file_name = f"pseudo_train_data.pt"

    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")
