import numpy as np
import random

def data_preprocess(in_path='../mpi3d_real.npz',
                    out_path='../mpi3d_fair.npz',
                    seed=1):

    random.seed(seed)

    # load the mpi3d_real dataset
    # the attributes are 
    # [object color, object shape, object size, 
    # camera height, background color, 
    # horizontal axis, vertical axis]
    data = np.load(in_path)['images']
    data = data.reshape([6,6,2,3,3,40,40,64,64,3])

    data = data[[0,2]] # object color: white=0 and red=1
    data = data[:,[1,4]] # object shape: cube=0 and pyramid=1
    # data = data[:,:,:] # size: small=0 and big=1
    data = data[:,:,:,:,[0,1]] # background color: purple=0 and sea green=1

    data = data.transpose([5,6,3,1,2,0,4,9,7,8])
    data = data.reshape([-1,2,2,2,2,3,64,64]) # [4800,2,2,2,2,3,64,64]
    # now attribute is [N, object shape, object size, object color, background color]

    # shuffle data
    N = data.shape[0]
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                for i4 in range(2):
                    order_list = list(range(N)) 
                    random.shuffle(order_list)
                    data[:,i1,i2,i3,i4] = data[order_list,i1,i2,i3,i4]

    # we will use 10% of data for training and 10% for testing
    # but in order to artificially creating correlation
    # we output 20%
    N_test = int(0.2 * data.shape[0])
    test_data = data[:N_test] # [960, 2, 2, 2, 2, 3, 64, 64]
    N_train = int(0.2 * data.shape[0])
    train_data = data[N_test:N_test+N_train] # [960, 2, 2, 2, 2, 3, 64, 64]

    np.savez(out_path,
        train_data=train_data, test_data=test_data)

if __name__ == '__main__':
    data_preprocess(
        in_path='mpi3d_real.npz',
        out_path='mpi3d_fair.npz'
        )