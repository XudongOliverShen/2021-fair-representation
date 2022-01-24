import torch
import torch.nn as nn
import numpy as np
import argparse
import math

def Balanced_MSE(recon, imgs, gender, weight):
    N = imgs.shape[0]
    recon_m = recon[gender==1]
    imgs_m = imgs[gender==1]
    recon_f = recon[gender==0]
    imgs_f = imgs[gender==0]
    loss_m = nn.MSELoss(reduction='sum')(recon_m, imgs_m)
    loss_f = nn.MSELoss(reduction='sum')(recon_f, imgs_f)
    loss = ((1-weight) * loss_m + weight * loss_f) / N
    return loss

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
        self.weights = self.get_weights()
    
    def scheduler_step(self):
        self.E_sched.step()
        self.D_sched.step()
    
    def get_weights(self):
        N = 0
        count = 0
        for idx, (h, labels) in enumerate(self.train_loader):
            N += h.shape[0]
            gender = labels[:,0]
            count += gender.sum().item()
        
        return count / N

    def train(self):

        self.Enc.train()
        self.Dec.train()

        for idx, (imgs, labels) in enumerate(self.train_loader):

            imgs = imgs.type(torch.float32).to(self.device)
            gender = labels[:,0].to(self.device)
            gender_float = gender.type(torch.float32).to(self.device)

            h = self.Enc(imgs)
            h_sens = torch.cat((h, gender_float.unsqueeze(-1)),1)
            recon = self.Dec(h_sens)
            l_I = self.l_I(recon, imgs)

            l_fair = self.l_fair(h, gender, self.args.fair_loss_alpha, self.args.fair_loss_l)
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
            for idx, (imgs, labels) in enumerate(self.eval_loader):

                imgs = imgs.type(torch.float32).to(self.device)
                gender = labels[:,0].to(self.device)
                gender_float = gender.type(torch.float32).to(self.device)

                h = self.Enc(imgs)
                h_sens = torch.cat((h, gender_float.unsqueeze(-1)),1)
                recon = self.Dec(h_sens)
                l_I = self.l_I(recon, imgs)

                l_fair = self.l_fair(h, gender, self.args.fair_loss_alpha, self.args.fair_loss_l)
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

