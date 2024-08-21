import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .augmentations import DataTransform
from sklearn.preprocessing import StandardScaler


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        # 传入的应该是一个电池列表，但是访问到具体的电池元素时，其本身就是一个字典

        # if len(X_train.shape) < 3:
        #     X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        #
        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)
        #
        # if isinstance(X_train, np.ndarray):
        #     self.x_data = torch.from_numpy(X_train)
        #     self.y_data = torch.from_numpy(y_train).long()
        # else:
        #     self.x_data = X_train
        #     self.y_data = y_train

        # self.len = X_train.shape[0]
        # nominal_capacity = 1.1
        scaler = StandardScaler()
        # labelset = [d['summary']['QD'][:] for d in dataset]
        labelset = [d['summary'] for d in dataset]
        labels = []
        # labelset = [d["summary"]/(d['summary'].max().max()) for d in dataset]
        data = [d['cycle'] for d in dataset]
        # dataset = [d["cycles"] for d in dataset]

        records = []
        ks = ['current_A', 'voltage_V', 'capacity_Ah', 'temperature_C']
        for i in range(len(dataset)):
            cell_data = data[i]
            cell_label = labelset[i]
            cycles = sorted(cell_data.keys(), key=lambda x: int(x))[:]
            labels.append(
                np.asarray(cell_label, dtype=np.float32)
            )
            records.append(
                np.asarray([[cell_data[c][k][:] for k in ks] for c in cycles],
                           dtype=np.float32)
            )
        del dataset

        num_samples = [len(d) for d in records]
        self._cum_sum = np.cumsum(num_samples)
        self.len = sum(num_samples)
        self.indexes = {}
        start = 0
        for i, s in enumerate(self._cum_sum):
            for idx in range(start, s):
                curr_idx = idx - start
                self.indexes[idx] = (i, curr_idx)
            start = s

        self.x_data = records
        self.y_data = labels

        for idx in range(len(self.x_data)):
            if isinstance(self.x_data[idx], np.ndarray):
                self.x_data[idx] = torch.from_numpy(self.x_data[idx])
                self.y_data[idx] = torch.from_numpy(self.y_data[idx])

        self.aug1 = []
        self.aug2 = []
        for idx in range(len(self.x_data)):
            # no need to apply Augmentations in other modes
            if training_mode == "self_supervised" or training_mode == "SupCon":
                a, b = DataTransform(self.x_data[idx], config)
                # a = torch.from_numpy(a)
                self.aug1.append(a)
                self.aug2.append(b)
        # if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
        #     self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        i, si = self.indexes[index]
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[i][si], self.y_data[i][si], self.aug1[i][si], self.aug2[i][si]
        else:
            return self.x_data[i][si], self.y_data[i][si], self.x_data[i][si], self.x_data[i][si]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    batch_size = configs.batch_size

    if "_1p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"))
    elif "_5p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
    elif "_10p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
    elif "_50p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
    elif "_75p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
    elif training_mode == "SupCon":
        train_dataset = torch.load(os.path.join(data_path, "pseudo_train_data.pt"))
    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))

    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    if train_dataset.__len__() < batch_size:
        batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader
