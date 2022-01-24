import numpy as np
from random import Random
import torch
from torch.utils.data import Dataset
import pickle
import os

def correlated_MPI3D_loaders(datapath, batch_size=128, DiBR=0, shuffle=True):
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
    train_dataset, test_dataset = correlated_MPI3D_datasets(datapath, DiBR)

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

def correlated_MPI3D_datasets(MPI3D, DiBR):

    # load data
    # the attributes are [object shape, object size, object color, background color]
    npzfile = np.load(MPI3D)
    train_data = npzfile['train_data'] # [960,2,2,2,2,3,64,64]
    test_data = npzfile['test_data'] # [960,2,2,2,2,3,64,64]

    # let's make object color the sensitive attribute, object shape the correlated attribute
    # the attributes are [object color, object shape, object size, background color]
    train_data = train_data.transpose([0,3,1,2,4,5,6,7])
    test_data = test_data.transpose([0,3,1,2,4,5,6,7])

    test_color = np.zeros(test_data.shape[:-3])
    test_color[:,1] = 1
    test_shape = np.zeros(test_data.shape[:-3])
    test_shape[:,:,1] = 1
    test_size = np.zeros(test_data.shape[:-3])
    test_size[:,:,:,1] = 1
    test_bgcolor = np.zeros(test_data.shape[:-3])
    test_bgcolor[:,:,:,:,1] = 1

    N1 = int(test_data.shape[0]* (0.5 + DiBR/2) )
    N2 = int(test_data.shape[0]* (0.5 - DiBR/2) )

    test_data = np.concatenate((test_data[:N1,0,0], test_data[:N2,0,1],
                                    test_data[:N2,1,0], test_data[:N1,1,1]), axis=0)
    test_color = np.concatenate((test_color[:N1,0,0], test_color[:N2,0,1],
                                    test_color[:N2,1,0], test_color[:N1,1,1]), axis=0)
    test_shape = np.concatenate((test_shape[:N1,0,0], test_shape[:N2,0,1],
                                    test_shape[:N2,1,0], test_shape[:N1,1,1]), axis=0)
    test_size = np.concatenate((test_size[:N1,0,0], test_size[:N2,0,1],
                                    test_size[:N2,1,0], test_size[:N1,1,1]), axis=0)
    test_bgcolor = np.concatenate((test_bgcolor[:N1,0,0], test_bgcolor[:N2,0,1],
                                    test_bgcolor[:N2,1,0], test_bgcolor[:N1,1,1]), axis=0)
    test_data = test_data.reshape([-1,3,64,64])
    test_color = test_color.reshape([-1])
    test_shape = test_shape.reshape([-1])
    test_size = test_size.reshape([-1])
    test_bgcolor = test_bgcolor.reshape([-1])

    test_dataset = MPI3D_dataset(test_data, test_color, test_shape, test_size, test_bgcolor)

    train_color = np.zeros(train_data.shape[:-3])
    train_color[:,1] = 1
    train_shape = np.zeros(train_data.shape[:-3])
    train_shape[:,:,1] = 1
    train_size = np.zeros(train_data.shape[:-3])
    train_size[:,:,:,1] = 1
    train_bgcolor = np.zeros(train_data.shape[:-3])
    train_bgcolor[:,:,:,:,1] = 1
    
    N1 = int(train_data.shape[0]* (0.5 + DiBR/2) )
    N2 = int(train_data.shape[0]* (0.5 - DiBR/2) )

    train_data = np.concatenate((train_data[:N1,0,0], train_data[:N2,0,1],
                                    train_data[:N2,1,0], train_data[:N1,1,1]), axis=0)
    train_color = np.concatenate((train_color[:N1,0,0], train_color[:N2,0,1],
                                    train_color[:N2,1,0], train_color[:N1,1,1]), axis=0)
    train_shape = np.concatenate((train_shape[:N1,0,0], train_shape[:N2,0,1],
                                    train_shape[:N2,1,0], train_shape[:N1,1,1]), axis=0)
    train_size = np.concatenate((train_size[:N1,0,0], train_size[:N2,0,1],
                                    train_size[:N2,1,0], train_size[:N1,1,1]), axis=0)
    train_bgcolor = np.concatenate((train_bgcolor[:N1,0,0], train_bgcolor[:N2,0,1],
                                    train_bgcolor[:N2,1,0], train_bgcolor[:N1,1,1]), axis=0)
    train_data = train_data.reshape([-1,3,64,64])
    train_color = train_color.reshape([-1])
    train_shape = train_shape.reshape([-1])
    train_size = train_size.reshape([-1])
    train_bgcolor = train_bgcolor.reshape([-1])
    
    train_dataset = MPI3D_dataset(train_data, train_color, train_shape, train_size, train_bgcolor)

    return train_dataset, test_dataset

class MPI3D_dataset(Dataset):
    def __init__(self, img, color, shape, size, bgcolor):
        self.img = torch.FloatTensor(img/255.0)
        self.color = torch.LongTensor(color)
        self.shape = torch.LongTensor(shape)
        self.size = torch.LongTensor(size)
        self.bgcolor = torch.LongTensor(bgcolor)

    def __getitem__(self, idx):
        return self.img[idx], self.color[idx], self.shape[idx], self.size[idx], self.bgcolor[idx]
    
    def __len__(self):
        return self.bgcolor.shape[0]
