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
        test_bat = [data['b1c4'], data['b1c7'], data['b1c8'], data['b1c14'],
                    data['b1c44']]
        train_val = []
        for idx, cn in enumerate(data):
            if data[cn] not in test_bat:
                train_val.append(data[cn])
        # 此处不需要split labels，因为labels是存在data内部的
        random.shuffle(train_val)
        train_len = int(len(train_val) * 0.80)

        return train_val[:train_len], train_val[train_len:], test_bat

    def save_dict(self, train_dict, val_dict, test_dict):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        train_file_path = os.path.join(save_path, 'train_1perc.pt')
        val_file_path = os.path.join(save_path, 'val_1perc.pt')
        test_file_path = os.path.join(save_path, 'test.pt')
        # note _use_new_zipfile_serialization will use zip_file format for storing data.
        torch.save(train_dict, train_file_path, _use_new_zipfile_serialization=False)
        torch.save(val_dict, val_file_path, _use_new_zipfile_serialization=False)
        torch.save(test_dict, test_file_path, _use_new_zipfile_serialization=False)
        print("pkl file has been split!")


current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
save_path = os.path.join(parent_path, 'data/cell_mit_batch1_1perc_data')
filepath = os.path.join(parent_path, 'data/batch1.pkl')

ld = Load_Dataset(save_path)
dataset = ld.read_pkl_file(filepath)
train_bat, val_bat, test_bat = ld.train_val_test_split(dataset)
ld.save_dict(train_bat, val_bat, test_bat)

