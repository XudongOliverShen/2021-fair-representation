import pandas
import os
from os.path import join as opj
from PIL import Image, ImageDraw
from os.path import join as opj
from tqdm import tqdm
import multiprocessing as mp
import pickle
from random import sample

######################################################################################################
# Step 1
######################################################################################################
bb_file = 'dataset/bb_landmark/loose_bb_train.csv'
lst_file = 'train.lst'
meta_file = 'dataset/identity_meta.csv'
num_parallel_process = 8

with open(meta_file) as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]

meta_info = []
count_train_ID = 0
count_test_ID = 0
for i in range(1, len(lines)):
    line = lines[i]
    name, _, _, flag, gender = line.split(', ')
    if gender == 'm':
        gender = 1
    elif gender =='f':
        gender = 0
    else:
        raise ValueError('gender is neither 1 nor 0!')

    if flag == '0':
        meta_info.append([name, 0, count_test_ID, gender])
        count_test_ID += 1
    elif flag == '1':
        meta_info.append([name, 1, count_train_ID, gender])
        count_train_ID += 1
    else: raise ValueError('aaa')

df = pandas.DataFrame(meta_info, columns=['ID', 'train_flag', 'label', 'gender'])

bb = pandas.read_csv(bb_file)
with open(lst_file, 'a+') as f:
    def single_process(i):
        NAME_ID = bb.iloc[i].loc['NAME_ID']
        ident = NAME_ID.split('/')[0]
        ident_label = int(df.loc[df.ID==ident].get('label'))
        gender = int(df.loc[df.ID==ident].get('gender'))
        path = NAME_ID + '.jpg'

        line = '\t'.join([str(i), str(ident_label), str(gender), path])
        line += '\n'
        f.write(line)

    pool = mp.Pool(processes=num_parallel_process)
    for _ in tqdm(pool.imap_unordered(single_process, range(bb.shape[0])), total=bb.shape[0]):
        pass


######################################################################################################
# Step 2
######################################################################################################
test_dir = 'test/'
lst_file = 'test_500x50.lst'

with open(meta_file) as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]

meta_info = []
count_train_ID = 0
count_test_ID = 0
for i in range(1, len(lines)):
    line = lines[i]
    name, _, _, flag, gender = line.split(', ')
    if gender == 'm':
        gender = 1
    elif gender =='f':
        gender = 0
    else:
        raise ValueError('gender is neither 1 nor 0!')

    if flag == '0':
        meta_info.append([name, 0, count_test_ID, gender])
        count_test_ID += 1
    elif flag == '1':
        meta_info.append([name, 1, count_train_ID, gender])
        count_train_ID += 1
    else: raise ValueError('aaa')

df = pandas.DataFrame(meta_info, columns=['ID', 'train_flag', 'label', 'gender'])
df = df[df.train_flag==0]

idx = 0
pp = 0
for ID in list(df.ID):

    gender = int(df[df.ID==ID].gender)

    dire = opj(test_dir, ID)
    imgs = os.listdir(dire)
    if len(imgs)<50:
        raise ValueError('aaa')
    imgs = sample(imgs, 50)
    for img in imgs:
        path = opj(ID, img)
        line = '\t'.join([str(idx), str(pp), str(gender), path])
        line += '\n'
        idx += 1
        with open(lst_file, 'a+') as f:
            f.write(line)
    
    pp += 1

print('finished!')