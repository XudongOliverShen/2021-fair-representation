import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import re
from datetime import datetime
from os.path import join as opj

import sys
sys.path.append("/home/e/e0474114/2020/FairEmbedding_final/MPI3D-E3")

from lib.utils import logger
from lib.MPI3D.MPI3D_dataset import correlated_MPI3D_loaders
from lib.fair_loss import MMD2_rq_u
from lib.MPI3D.net import ResNet34, ResNet34_transposed
from lib.MPI3D.process_train import process_train
    
def parse_option():

    parser = argparse.ArgumentParser(description='training encoders....', fromfile_prefix_chars='@')
    parser.add_argument('--dataset', type=str,
                            default='data/MPI3D/mpi3d_fair.npz')
                            # default='/home/e/e0474114/2020/FairEmbedding_final/MPI3D-E2/data/mpi3d_fair.npz')
    parser.add_argument('--correlation', type=float, default=0,
                            help='level of correlation, between 0 and 1')
    parser.add_argument('--out_dir', type=str,
                            default='checkpoints/MPI3D')
                            # default='/home/e/e0474114/2020/FairEmbedding_final/2021-fair-representation/checkpoints/MPI3D')

    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--batch_size', type=int, default=256,
                            help='batch size')
    parser.add_argument('--epochs', type=int, default=150,
                            help='number of epochs')
    parser.add_argument('--scheduler_steps', type=str, default='80,110,130',
                            help='scheduler step size')
    parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,
                            help='Learning rate step gamma')
    
    parser.add_argument('--dim', type=int, default=32,
                            help='representation dimension')
    parser.add_argument('--lmd', type=float, default=10)
    parser.add_argument('--fair_loss_alpha', type=float, default=4)
    parser.add_argument('--fair_loss_l', type=float, default=1)

    parser.add_argument('--visdom_name', type=str, default='ResNet-34')
    parser.add_argument('--visdom_port', type=int, default=8099)
    args= parser.parse_args()
    return args

def main():
    args = parse_option()
    args.out_dir = os.path.join(args.out_dir)
    args.scheduler_steps = [int(i) for i in args.scheduler_steps.split(',')]

    loaders = correlated_MPI3D_loaders(args.dataset, args.batch_size, args.correlation)

    # save dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    Logger = logger(args.visdom_name, os.path.join(args.out_dir, 'training_stats.pkl'), args.visdom_port)


    fair_loss_func = MMD2_rq_u
    
    Enc = ResNet34(dim_out = args.dim).to(args.device)
    Dec = ResNet34_transposed(dim_in = args.dim * 2).to(args.device) # dim_in = args.dim * 2 is a hack
    process = process_train(Enc = Enc,
                    Dec = Dec,
                    pretext_loss = nn.MSELoss(),
                    fair_loss = fair_loss_func,
                    loaders = loaders,
                    logger = Logger,
                    args=args)

    for epoch in range(1,args.epochs+1):
        process.train()
        process.eval()
        process.scheduler_step()

    Logger.reset_each_run(1)
    torch.save(process.Enc.state_dict(), opj(args.out_dir,'encoder.pt'))

if __name__ == '__main__':
    main()
