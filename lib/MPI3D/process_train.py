import torch
import torch.nn as nn
import numpy as np
import argparse
import math

def WKNN_evaluation(h, labels, k = 1):
    """ weighted k-nearest neighbor evaluation

    predict by weighted majrotiy vote of the k-nearest neighbors' labels

    Args:
        h (torch tensor, [N, d]):
        labels (torch tensor, [N, 4]):
        k:

    Returns:
        acc (torch tensor, [4]):
    """
    N, M = labels.shape
    BR = labels.sum(dim=0) / N # base rates

    preds = torch.empty( (N, M), device=labels.device)
    for i in range(N):
        dist = np.array((h - h[i]).norm(dim=-1).cpu())
        idx = set(np.argpartition(dist, k+1)[:k+1])
        idx.remove(i)

        if len(idx) == k:
            pred = labels[list(idx)]
        else:
            raise ValueError('Didnt find exactly {} nearest neighbor!'.format(k))

        preds[i] = pred.sum(dim=0) / k

    # calculate Balanced Accuracy, BAcc
    matrix = (preds >= BR).float() == labels
    BAcc = torch.empty( (M), device=labels.device)
    for i in range(M):
        N_pos = (labels[:,i]==1).sum().item()
        N_neg = (labels[:,i]==0).sum().item()
        BAcc[i] = 0.5 * matrix[labels[:,i]==1, i].sum() / N_pos + 0.5 * matrix[labels[:,i]==0, i].sum() / N_neg

    # calculate Average Accuracy, AAcc
    matrix = (preds >= BR[0].item()).float() == labels
    N_pos = (labels[:,0]==1).sum().item()
    N_neg = (labels[:,0]==0).sum().item()
    AAcc = 0.5 * matrix[labels[:,0]==1].sum(dim=0) / N_pos + 0.5 * matrix[labels[:,0]==0].sum(dim=0) / N_neg
        
    return BAcc, AAcc

class process_train(object):
    def __init__(self, Enc, Dec, pretext_loss, fair_loss, loaders, logger, args):
        self.Enc = Enc
        self.Dec = Dec
        self.E_opt = torch.optim.SGD(self.Enc.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
        self.D_opt = torch.optim.SGD(self.Dec.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
        self.E_sched = torch.optim.lr_scheduler.MultiStepLR(self.E_opt, milestones=args.scheduler_steps, gamma=args.gamma)
        self.D_sched = torch.optim.lr_scheduler.MultiStepLR(self.D_opt, milestones=args.scheduler_steps, gamma=args.gamma)

        self.l_I = pretext_loss
        self.l_fair = fair_loss

        self.train_loader, self.eval_loader = loaders
        self.logger = logger

        self.lmd = args.lmd
        self.device = args.device
        self.args = args
    
    def scheduler_step(self):
        self.E_sched.step()
        self.D_sched.step()

    def train(self):

        self.Enc.train()
        self.Dec.train()

        for idx, (imgs, bg_color, shape, size, color) in enumerate(self.train_loader):

            imgs = imgs.to(self.device)
            bg_color = bg_color.to(self.device)
            bg_color_float = bg_color.type(imgs.type()).detach()

            h = self.Enc(imgs)
            h_sens = torch.cat((h, bg_color_float.unsqueeze(-1).repeat(1,h.shape[1])),1)
            recon = self.Dec(h_sens)
            l_I = self.l_I(recon, imgs)

            l_fair = self.l_fair(h, bg_color, self.args.fair_loss_alpha, self.args.fair_loss_l)
            loss = 1/(1+self.lmd) * l_I + self.lmd/(1+self.lmd) * l_fair

            self.E_opt.zero_grad()
            self.D_opt.zero_grad()
            loss.backward()
            self.E_opt.step()
            self.D_opt.step()
        
        self.logger.log('train', 'loss', 'recon_loss', l_I.item())
        self.logger.log('train', 'loss', 'fair_loss', l_fair.item())
        self.logger.log('train', 'loss', 'loss', loss.item())
    
    def eval(self):

        self.Enc.eval()
        self.Dec.eval()

        l_I_sum = 0
        l_fair_sum = 0
        loss_sum = 0

        with torch.no_grad():
            for idx, (imgs, bg_color, shape, size, color) in enumerate(self.eval_loader):

                imgs = imgs.to(self.device)
                bg_color = bg_color.to(self.device)
                shape = shape.to(self.device)
                size = size.to(self.device)
                color = color.to(self.device)
                bg_color_float = bg_color.type(imgs.type()).detach()

                h = self.Enc(imgs)
                h_sens = torch.cat((h, bg_color_float.unsqueeze(-1).repeat(1,h.shape[1])),1)
                recon = self.Dec(h_sens)

                l_I = self.l_I(recon, imgs)
                l_fair = self.l_fair(h, bg_color, self.args.fair_loss_alpha, self.args.fair_loss_l)
                loss = 1/(1+self.lmd) * l_I + self.lmd/(1+self.lmd) * l_fair

                n = imgs.shape[0]
                l_I_sum += l_I.item() * n
                l_fair_sum += l_fair.item() * n
                loss_sum += loss.item() * n
        
            N = self.eval_loader.dataset.__len__()
            l_I = l_I_sum / N
            l_fair = l_fair_sum / N
            loss = loss_sum / N

            self.logger.log('test', 'loss', 'recon_loss', l_I)
            self.logger.log('test', 'loss', 'fair_loss', l_fair)
            self.logger.log('test', 'loss', 'loss', loss)

