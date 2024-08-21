import pickle
from sklearn.model_selection import train_test_split
import os
import torch
import random

class Load_Dataset():
    def __init__(self, save_path):
        super(Load_Dataset).__init__()
        self.save_path = save_path

    def read_pkl_file(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def train_val_test_split(data):
        # we simulate article CL in battery, pick c-4, c-7, c-8, c-14, c-44
        # as test bat
        # test_bat = [data['b1c4'], data['b1c7'], data['b1c8'], data['b1c14'],
        #             data['b1c44']]
        test_bat_idx = [3, 6, 7, 13, 43]
        train_val = []
        test_bat = []
        for idx in range(len(data)):
            if idx not in test_bat_idx:
                train_val.append(data[idx])
        # 此处不需要split labels，因为labels是存在data内部的
        random.shuffle(train_val)
        train_len = int(len(train_val) * 0.80)
        for idx in test_bat_idx:
            test_bat.append(data[idx])
        return train_val[:train_len], train_val[train_len:], test_bat

    def percent_data(self, train_data, percentage):
        labels = []  # 用labels的idx来作为电池的标签
        train_copy = train_data.copy()
        for bat in train_data:
            labels.append(bat['summary']['QD'][:])

        for bat in range(len(labels)):
            indexes = list(range(len(labels[bat])))
            random.shuffle(indexes)
            zp = int(len(indexes) * (1 - percentage))
            train_copy[bat]['summary']['QD'] = train_copy[bat]['summary']['QD'][indexes[zp:]]
            for zi in indexes[:zp]:
                del train_copy[bat]['cycles'][str(zi)]  # 将选取到的Index全部删除

        return train_copy


    def save_dict(self, train_dict, val_dict, test_dict):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        train_file_path = os.path.join(save_path, 'train.pt')
        val_file_path = os.path.join(save_path, 'val.pt')
        test_file_path = os.path.join(save_path, 'test.pt')
        # note _use_new_zipfile_serialization will use zip_file format for storing data.
        torch.save(train_dict, train_file_path, _use_new_zipfile_serialization=False)
        torch.save(val_dict, val_file_path, _use_new_zipfile_serialization=False)
        torch.save(test_dict, test_file_path, _use_new_zipfile_serialization=False)
        print("pkl file has been split!")

    def save_one_dict(self, train_dict):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        train_file_path = os.path.join(save_path, 'train_10perc.pt')
        # note _use_new_zipfile_serialization will use zip_file format for storing data.
        torch.save(train_dict, train_file_path, _use_new_zipfile_serialization=False)
        print("pkl file has been saved!")


current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
save_path = os.path.join(parent_path, 'data/cell_mit_batch3_prepocess_data')
filepath = os.path.join(parent_path, 'data/batch3_prepocess.pkl')

ld = Load_Dataset(save_path)
dataset = ld.read_pkl_file(filepath)
train_bat, val_bat, test_bat = ld.train_val_test_split(dataset)
# train_bat_10perc = ld.percent_data(train_bat, 0.1)
# ld.save_one_dict(train_bat_10perc)
ld.save_dict(train_bat, val_bat, test_bat)

