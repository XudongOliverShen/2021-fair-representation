import numpy as np
from random import Random
import torch
from torch.utils.data import Dataset
import pickle
import os


def Adult_loaders(datapath, batch_size=128, shuffle=True):
    """ create a correlated MPI3D data loader

    we assume the sum of the base rates is 1.
    the difference in base rates (DiBR) controls the level of correlation.
    no correlation if DiBR is 0 and maximum correlation if DiBR is 1.

    Args:
        datapath (str):
        batch_size (int):
        DiBR (float): difference in base rates, [0,1]
        shuffle (bool):

    Returns:
        train_loader, test_loader
    """

    # dataset
    train_dataset, test_dataset = Adult_datasets(datapath)

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=shuffle
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=shuffle
    )

    return train_loader, test_loader

def Adult_datasets(MPI3D):

    # load data
    # label [gender, income class]
    npzfile = np.load(MPI3D)
    train_data = npzfile['train_data']
    train_label = npzfile['train_label']
    test_data = npzfile['test_data']
    test_label = npzfile['test_label']

    train_dataset = Adult_dataset(train_data, train_label)
    test_dataset = Adult_dataset(test_data, test_label)

    return train_dataset, test_dataset

class Adult_dataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return self.data.shape[0]


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='# of epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    