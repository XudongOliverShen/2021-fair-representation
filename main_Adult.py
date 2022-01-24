import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import re
from datetime import datetime
from os.path import join as opj


from lib.Adult.Adult_dataset import Adult_loaders
from lib.utils import logger
from lib.fair_loss import MMD2_rq_u
from lib.Adult.net import DenseBlock
from lib.Adult.process_train import process_train
    
def parse_option():

    parser = argparse.ArgumentParser(description='training encoders....', fromfile_prefix_chars='@')
    parser.add_argument('--dataset', type=str,
                            # default='data/Adult/Adult_fair.npz')
                            default='/home/e/e0474114/2020/FairEmbedding_final/Adult-E1/data/Adult_fair.npz')
    parser.add_argument('--out_dir', type=str,
                            # default='checkpoints/Adult')
                            default='/home/e/e0474114/2020/FairEmbedding_final/2021-fair-representation/checkpoints/Adult')

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=256,
                            help='batch size')
    parser.add_argument('--epochs', type=int, default=110,
                            help='number of epochs')
    parser.add_argument('--scheduler_steps', type=str, default='70,90',
                            help='scheduler step size')
    parser.add_argument('--lr', type=float, default=1,
                            help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,
                            help='Learning rate step gamma')
    
    parser.add_argument('--dim', type=int, default=16,
                            help='representation dimension')
    parser.add_argument('--lmd', type=float, default=200)
    parser.add_argument('--fair_loss_alpha', type=float, default=2)
    parser.add_argument('--fair_loss_l', type=float, default=2)

    parser.add_argument('--visdom_name', type=str, default='DenseNet')
    parser.add_argument('--visdom_port', type=int, default=8099)
    args= parser.parse_args()
    return args

def main():
    args = parse_option()
    args.out_dir = os.path.join(args.out_dir)
    args.scheduler_steps = [int(i) for i in args.scheduler_steps.split(',')]

    loaders = Adult_loaders(args.dataset, args.batch_size, True)

    # save dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    Logger = logger(args.visdom_name, os.path.join(args.out_dir, 'training_stats.pkl'), args.visdom_port)

    fair_loss_func = MMD2_rq_u

    Enc = DenseBlock(dim_in=166, dim_hidden=166*8, dim_out=args.dim, sigmoid=False).to(args.device)
    Dec = DenseBlock(dim_in=args.dim+1, dim_hidden=166*8, dim_out=166, sigmoid=True).to(args.device)
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
