import pandas
import os
from os.path import join as opj
from PIL import Image, ImageDraw
from os.path import join as opj
from tqdm import tqdm
import multiprocessing as mp
import pickle

lst_file = 'test.lst'
data_dir = 'test/'

pp_list = os.listdir(data_dir)

img_count = 0
for i in range(len(pp_list)):
    pp = pp_list[i]
    pp_dir = opj(data_dir, pp)
    for img_name in os.listdir(pp_dir):
        img_path = opj(pp, img_name)
        line = '\t'.join([str(img_count), str(i), img_path])
        line += '\n'
        with open(lst_file, 'a+') as f:
            f.write(line)
        img_count += 1

