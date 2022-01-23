import os
from os.path import join as opj
import argparse
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import re
from datetime import datetime
from os.path import join as opj

from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torchvision.transforms as transforms

from lib.utils import VisdomLinePlotter, logger, MyPipeline
from lib.VGGFace2.LFW_dataset import LFWDataset
from lib.VGGFace2.net import sphere
from lib.VGGFace2.process_train import process_train
from lib.VGGFace2.face_loss import ArcFace
from lib.fair_loss import MMD2_rq_u

def parse_option():

    parser = argparse.ArgumentParser(description='training encoders....', fromfile_prefix_chars='@')
    parser.add_argument('--VGGFace2_dir', type=str,
                            default='data/VGGFace2_112_112')
                            # default='/temp/xudong/VGGFace2_112_112_dataset')
    parser.add_argument('--LFW_dir', type=str,
                            default='data/LFW_112_112')
                            # default='/temp/xudong/LFW_112_112_dataset')
    parser.add_argument('--pkl_dir', type=str,
                            default='checkpoints/VGGFace2')    
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', type=int, default=256,
                            help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                            help='number of epochs')
    parser.add_argument('--scheduler_steps', type=str, default='140,180',
                            help='scheduler step size')
    parser.add_argument('--lr', type=float, default=1e-4,
                            help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,
                            help='Learning rate step gamma')
    
    parser.add_argument('--emd_dim', type=int, default=32)
    parser.add_argument('--lmd1', type=float, default=3e1)
    parser.add_argument('--lmd2', type=float, default=3e1)
    parser.add_argument('--fair_loss_coef', type=float, default=0.5)
    parser.add_argument('--fair_loss_alpha', type=float, default=4)
    parser.add_argument('--fair_loss_l', type=float, default=1)

    parser.add_argument('--visdom_name', type=str, default='sphereNet20')
    args = parser.parse_args()
    return args

def main():

    args = parse_option()
    args.scheduler_steps = [int(i) for i in args.scheduler_steps.split(',')]

    print('building pipelines...')
    train_pipe = MyPipeline(rec_file=opj(args.VGGFace2_dir, 'train.rec'),
                            idx_file=opj(args.VGGFace2_dir, 'train.idx'),
                            batch_size=args.batch_size,
                            num_threads=4,
                            device_id=int(args.device.split(':')[1]),
                            shuffle=True)
    test_pipe = MyPipeline(rec_file=opj(args.VGGFace2_dir, 'test_500x50.rec'),
                            idx_file=opj(args.VGGFace2_dir, 'test_500x50.idx'),
                            batch_size=args.batch_size,
                            num_threads=4,
                            device_id=int(args.device.split(':')[1]),
                            shuffle=False)

    LFW_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.5, 0.5],
            std  = [0.5, 0.5, 0.5]
        )
    ])
    LFW_dataloader = torch.utils.data.DataLoader(
        dataset = LFWDataset(dir = opj(args.LFW_dir, 'test'),
                        pairs_path = opj(args.LFW_dir, 'pairs.txt'),
                        transform = LFW_transforms
                        ),
        batch_size = args.batch_size,
        num_workers = 4,
        shuffle = False
    )

    train_pipe.build()
    test_pipe.build()

    # provide reader_name so that iter stop at the end of every epoch
    # auto_reset reset the iterator after every epoch
    train_iter = DALIGenericIterator(train_pipe, ['imgs', 'labels'],
                                        size = 512*500,
                                        # reader_name='Reader',
                                        auto_reset=True)
    test_iter = DALIGenericIterator(test_pipe, ['imgs', 'labels'],
                                        reader_name='Reader',
                                        auto_reset=True,
                                        fill_last_batch=False)

    print('finished building pipelines!')

    # save dir
    if not os.path.exists(args.pkl_dir):
        os.mkdir(args.pkl_dir)

    Logger = logger(args.visdom_name, os.path.join(args.pkl_dir, 'training_stats.pkl'))

    fair_loss_func = MMD2_rq_u
    face_loss = ArcFace(emdsize=args.emd_dim+1, class_num=8631, s=64, m=.3, device=args.device, lst_file=opj(args.VGGFace2_dir, 'train.lst'))

    model = sphere(type=20, is_gray=False, emd_dim=args.emd_dim).to(args.device)
    process = process_train(model, train_iter, test_iter, LFW_dataloader, face_loss, fair_loss_func, Logger, args)

    for epoch in range(1,args.epochs+1):
        process.train()
        process.test()
        process.scheduler.step()

    Logger.reset_each_run(1) # This save training statistics
    torch.save(model.state_dict(), opj(args.pkl_dir,'encoder.pt'))

if __name__ == '__main__':
    main()