import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os

class Load_Dataset():
    def __init__(self, normalization_method, save_path, file_list):
        super(Load_Dataset, self).__init__()
        self.normalization_method = normalization_method
        self.filelist = file_list
        self.save_path = save_path


    def read_one_csv(self, filename, nominal_capacity = None):
        df = pd.read_csv(filename)
        df.insert(df.shape[1]-1, 'cycle index', np.arange(df.shape[0]))

        if nominal_capacity is not None:
            df['capacity'] = df['capacity']/nominal_capacity
            f_df = df.iloc[:, :-1]
            if self.normalization_method == 'min-max':
                f_df = 2*(f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()
            df.iloc[:, :-1] = f_df
        return df

    def load_one_battery(self, filename, nominal_capacity=None):
        df = self.read_one_csv(filename, nominal_capacity)
        x_total = df.iloc[:, :-1].values
        y_total = df.iloc[:, -1].values
        x = x_total[:-1]
        y = y_total[:-1]
        return x, y

    def load_all_battery(self, nominal_capacity=None):
        battery_data_list = []
        battery_labels_list = []
        for path in self.filelist:
            (x, y) = self.load_one_battery(path, nominal_capacity)
            battery_data_list.append(x)
            battery_labels_list.append(y)

        # battery_data_list = np.concatenate(battery_data_list, axis=0)
        # battery_labels_list = np.concatenate(battery_labels_list, axis=0)

        # battery_data_list.reshape(battery_data_list.shape[0], 1, battery_data_list.shape[1])

        return battery_data_list, battery_labels_list

    @staticmethod
    def train_val_split(dataset, labels, train_factor):
        split = int(len(dataset) * train_factor)
        train_val_x, test_x = dataset[:split], dataset[split:]
        train_val_y, test_y = labels[:split], labels[split:]

        train_x, valid_x, train_y, valid_y = train_test_split(train_val_x, train_val_y, test_size=0.2,
                                                              shuffle=False)
        train_dict = {"samples": train_x, "labels": train_y}
        val_dict = {"samples": valid_x, "labels": valid_y}
        test_dict = {"samples": test_x, "labels": test_y}

        return train_dict, val_dict, test_dict

    def save_dict(self, train_dict, val_dict, test_dict):
        savepath = self.save_path
        os.makedirs(savepath, exist_ok=True)
        train_file_path = os.path.join(savepath, 'train.pt')
        val_file_path = os.path.join(savepath, 'val.pt')
        test_file_path = os.path.join(savepath, 'test.pt')
        torch.save(train_dict, train_file_path)
        torch.save(val_dict, val_file_path)
        torch.save(test_dict, test_file_path)
        print("Dataset has been split!")


current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
save_path = os.path.join(parent_path, 'data/cell_mit_batch1')
fileroot = os.path.join(parent_path, 'data/MIT data/2017-05-12')
file_list = os.listdir(fileroot)
file_list_path = []
for file in file_list:
    file_name = os.path.join(fileroot, file)
    file_list_path.append(file_name)
nominal_capacity = 1.1
normalization_method = 'z-score'

ld = Load_Dataset(normalization_method, save_path=save_path, file_list=file_list_path)
dataset, labels = ld.load_all_battery(nominal_capacity=1.1)
train_dict, val_dict, test_dict = ld.train_val_split(dataset, labels, 0.9)
ld.save_dict(train_dict, val_dict, test_dict)


