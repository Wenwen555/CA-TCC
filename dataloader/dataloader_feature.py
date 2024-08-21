import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
from sklearn.model_selection import train_test_split
# from utils.util import write_to_txt
from models.test_model import Encode_Model
import argparse
from .augmentations import DataTransform
from torch.utils.data import Dataset
# 传入参数的不同： 数据集、训练模式


class DF():
    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method  # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self, df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_one_csv(self,file_name, nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)
        df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        if nominal_capacity is not None:
            #print(f'nominal_capacity:{nominal_capacity}, capacity max:{df["capacity"].max()}',end=',')
            df['capacity'] = df['capacity']/nominal_capacity
            #print(f'SOH max:{df["capacity"].max()}')
            f_df = df.iloc[:, :-1]
            if self.normalization_method == 'min-max':
                f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()

            df.iloc[:,:-1] = f_df

        return df

    def load_one_battery(self,path,nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        df = self.read_one_csv(path,nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1 = x[:-1]
        # x2 = x[1:]
        y1 = y[:-1]
        # y2 = y[1:]
        # return (x1,y1),(x2,y2)
        return (x1, y1)

    def load_all_battery(self,path_list,nominal_capacity):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :param batch_size: batch size
        :return: Dataloader
        '''
        # X1, X2, Y1, Y2 = [], [], [], []
        X1, Y1 = [], []
        # if self.args.log_dir is not None and self.args.save_folder is not None:
        #     save_name = os.path.join(self.args.save_folder,self.args.log_dir)
        #     write_to_txt(save_name,'data path:')
        #     write_to_txt(save_name,str(path_list))
        for path in path_list:
            # (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            (x1, y1) = self.load_one_battery(path, nominal_capacity)
            # print(path)
            # print(x1.shape, x2.shape, y1.shape, y2.shape)
            X1.append(x1)
            # X2.append(x2)
            Y1.append(y1)
            # Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        # X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        # Y2 = np.concatenate(Y2, axis=0)


        tensor_X1 = torch.from_numpy(X1).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)
        tensor_X1 = tensor_X1.unsqueeze(1)  # 将tensor化为[samples_num, 1 = one cycle, feature_length]

        # tensor_X2 = torch.from_numpy(X2).float()

        # tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)
        # print('X shape:',tensor_X1.shape)
        # print('Y shape:',tensor_Y1.shape)

        # 有时候需要指定训练集和测试集的电池ID，因此这个函数返回一个字典，里面包含多种情况，
        # 可根据需要选择
        # 1. 传入的path_list是【训练集】、【验证集】和【测试集】的电池ID，这时候按照前80%训练，后20%测试划分，再从训练集中随机划分出验证集，比例为8:2
        # 2. 传入的path_list是【训练集】和【测试集】的电池ID，这种情况只需要按照8:2随机化划分训练集和测试集即可
        # 3. 传入的path_list是【测试集】的电池ID，则不需要划分，直接封装成dataloader即可
        ## English version
        # Sometimes it is necessary to specify the battery ID of the training set and test set,
        # so this function returns a dictionary containing a variety of situations,
        # You can choose according to your needs
        # 1. The incoming path_list is the battery ID of [training set], [validation set] and [test set].
        #     At this time, it is divided into the first 80% for training and the last 20% for testing,
        #     and then the validation set is randomly divided from the training set. The ratio is 8:2
        # 2. The incoming path_list is the battery ID of [training set] and [testing set].
        #     In this case, you only need to randomly divide the training set and test set according to 8:2.
        # 3. The incoming path_list is the battery ID of the [test set], so there is no need to divide it and it can be directly encapsulated into a dataloader.

        # Condition 1
        # 1.1 划分训练集和测试集
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        # train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        # train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]

        # 1.2 划分训练集和验证集
        # train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
        #     train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

        train_X1, valid_X1, train_Y1, valid_Y1 = \
            train_test_split(train_X1, train_Y1, test_size=0.2, random_state=420)

        train_Y1 = train_Y1.squeeze(1)
        valid_Y1 = valid_Y1.squeeze(1)
        test_Y1 = test_Y1 .squeeze(1)

        train_dict = {"samples": train_X1, "labels": train_Y1}
        val_dict = {"samples": valid_X1, "labels": valid_Y1}
        test_dict = {"samples": test_X1, "labels": test_Y1}

        return train_dict, val_dict, test_dict

        # train_loader = DataLoader(TensorDataset(train_X1, train_Y1),
        #                           batch_size=self.args,
        #                           shuffle=True)
        # valid_loader = DataLoader(TensorDataset(valid_X1, valid_Y1),
        #                           batch_size=self.args.batch_size,
        #                           shuffle=True)
        # test_loader = DataLoader(TensorDataset(test_X1, test_Y1),
        #                          batch_size=self.args.batch_size,
        #                          shuffle=False)

        # train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
        #                           batch_size=self.args.batch_size,
        #                           shuffle=True)
        # valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
        #                           batch_size=self.args.batch_size,
        #                           shuffle=True)
        # test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
        #                          batch_size=self.args.batch_size,
        #                          shuffle=False)


        # Condition 2
        # train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
        #     train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
        # train_loader_2 = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
        #                           batch_size=self.args.batch_size,
        #                           shuffle=True)
        # valid_loader_2 = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
        #                           batch_size=self.args.batch_size,
        #                           shuffle=True)
        #
        # # Condition 3
        # test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
        #                          batch_size=self.args.batch_size,
        #                          shuffle=False)

        #
        # loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
        #           'train_2': train_loader_2,'valid_2': valid_loader_2,
        #           'test_3': test_loader_3}

        # return train_loader, valid_loader, test_loader
        # return loader

class MITdata(DF):
    def __init__(self, root='data/MIT data', args=None):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12', '2017-06-30', '2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-' * 20, 'MIT data', '-' * 20)

    def read_one_batch(self, batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]
        :return: dict
        '''
        assert batch in [1, 2, 3], 'batch must be in {}'.format([1, 2, 3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件。如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file; otherwise read all files;
        :param self:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root,batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root,file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode):
        super(load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        self.x_data = X_train
        self.y_data = y_train

        self.len = X_train.shape[0]

        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='MIT', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch', type=int, default=1, help='1,2,3')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
    parser.add_argument('--log_dir', type=str, default='test.txt', help='log dir')
    return parser.parse_args()


def data_generator(configs, training_mode):
    batch_size = configs.batch_size
    class Args:
        pass
    args = Args()
    args.data = "MIT"
    args.batch = 1
    args.batch_size = 256
    args.normalization_method = 'z-score'
    args.log_dir = 'test.txt'
    mit = MITdata(args=args)
    train_dict, val_dict, test_dict = mit.read_one_batch(batch=1)

    train_dataset = load_Dataset(train_dict, configs, training_mode)
    valid_dataset = load_Dataset(val_dict, configs, training_mode)
    test_dataset = load_Dataset(test_dict, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader

    # current_path = os.getcwd()
    # parent_path = os.path.dirname(current_path)
    # folder_path = os.path.join(parent_path, 'data', 'cell_mit_batch1')

    # 创建文件夹，如果不存在则创建
    # os.makedirs(folder_path, exist_ok=True)
    #
    # # 定义文件路径
    # train_file_path = os.path.join(folder_path, 'train.pt')
    # val_file_path = os.path.join(folder_path, 'val.pt')
    # test_file_path = os.path.join(folder_path, 'test.pt')
    #
    # # 保存字典到 .pt 文件
    # torch.save(train_dict, train_file_path)
    # torch.save(val_dict, val_file_path)
    # torch.save(test_dict, test_file_path)
    #
    # print(f"train_dict saved to {train_file_path}")
    # print(f"val_dict saved to {val_file_path}")
    # print(f"test_dict saved to {test_file_path}")
    # loader = mit.read_all()
