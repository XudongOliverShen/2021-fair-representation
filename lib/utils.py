from visdom import Visdom
import numpy as np
from random import Random
import torch
from torch.utils.data import Dataset
import pickle
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class MyPipeline(Pipeline):
    def __init__(self, rec_file, idx_file, batch_size, num_threads, device_id, shuffle):

        with open(idx_file, 'rb') as f:
            lines = f.readlines()
        num_records = len(lines)
        self.dataset_size = num_records

        super(MyPipeline, self).__init__(batch_size, num_threads, device_id, seed=num_records)
        self.input = ops.MXNetReader(path=rec_file, index_path=idx_file,
                                    initial_fill=num_records, random_shuffle = shuffle)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.MirrorNormalize = ops.CropMirrorNormalize(
                                                device='gpu',
                                                mean=[127.5, 127.5, 127.5],
                                                std = [127.5, 127.5, 127.5]
                                                )
        self.coin = ops.CoinFlip(probability=0.5)
    
    def define_graph(self):
        rdn = self.coin()
        imgs, labels = self.input(name='Reader')
        imgs = self.decode(imgs)
        imgs = self.MirrorNormalize(imgs, mirror=rdn)
        return (imgs, labels)

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
    
class logger(object):

    def __init__(self, env_name, pkl_path):
        
        self.dict = {}
        self.training_stat = {}
        self.plotter = VisdomLinePlotter(env_name=env_name)
        self.dict_path = pkl_path
    
    def log(self, state, var, name, val):

        if state not in self.training_stat.keys():
            self.training_stat[state] = {}
        if var not in self.training_stat[state].keys():
            self.training_stat[state][var] = {}
        if name not in self.training_stat[state][var].keys():
            self.training_stat[state][var][name] = []
        
        self.training_stat[state][var][name].append(val)
        self.plotter.plot(name, state, var+' | '+name, len(self.training_stat[state][var][name]), val)

    def reset_each_run(self, i_th):
        if not os.path.exists(self.dict_path):
            save_dict = {}
            save_dict[i_th] = self.training_stat.copy()
            with open(self.dict_path, 'wb') as f:
                pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.dict_path, 'rb') as f:
                save_dict = pickle.load(f)
            save_dict[i_th] = self.training_stat.copy()
            with open(self.dict_path, 'wb') as f:
                pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
        self.training_stat.clear()
