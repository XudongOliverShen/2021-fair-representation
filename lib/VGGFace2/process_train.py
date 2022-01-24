import torch
import torch.nn as nn
import numpy as np
import argparse
import math

# from utils import VisdomLinePlotter, logger
from tqdm import tqdm
from .LFW_evaluate import evaluate_lfw

def polarize(h):
    v = torch.acos(h[:,0]/h[:,0:].norm(dim=-1)).unsqueeze(dim=1)
    for i in range(1, h.shape[1]-2):
        vnew = torch.acos(h[:,i]/h[:,i:].norm(dim=-1))
        v = torch.cat([v,vnew.unsqueeze(dim=1)], dim=1)
    idx = h[:,-1] >= 0
    vnew[idx] = torch.acos(h[idx,-2]/h[idx,-2:].norm(dim=-1))
    idx = h[:,-1] < 0
    vnew[idx] = 2*math.pi - torch.acos(h[idx,-2]/h[idx,-2:].norm(dim=-1))
    v = torch.cat([v,vnew.unsqueeze(dim=1)], dim=1)
    return v

def eval_identification(similarity, labels):
    N = similarity.shape[0]
    R1_correct = 0
    R1_incorrect = 0
    R10_correct = 0
    R10_incorrect = 0
    R100_correct = 0
    R100_incorrect =0
    for i_probe in range(N):
        same_pp = set((labels==labels[i_probe]).nonzero().squeeze().tolist())
        same_pp.discard(i_probe)
        sim_2_gallery = similarity[list(same_pp),:][:, labels!=labels[i_probe]]
        sim_2_probe = similarity[i_probe, list(same_pp)]

        sim_2_gallery, _ = torch.sort(sim_2_gallery, dim=-1, descending=True)

        thr = sim_2_gallery[:,0]
        indicator = (sim_2_probe > thr)
        R1_correct += (indicator==True).sum().item()
        R1_incorrect += (indicator==False).sum().item()

        thr = sim_2_gallery[:,9]
        indicator = (sim_2_probe > thr)
        R10_correct += (indicator==True).sum().item()
        R10_incorrect += (indicator==False).sum().item()

        thr = sim_2_gallery[:,99]
        indicator = (sim_2_probe > thr)
        R100_correct += (indicator==True).sum().item()
        R100_incorrect += (indicator==False).sum().item()
    
    return R1_correct/(R1_correct + R1_incorrect), R10_correct/(R10_correct + R10_incorrect), R100_correct/(R100_correct + R100_incorrect)


def eval_verification(similarity, labels):
    N = similarity.shape[0]

    match_similarity = []
    non_match_similarity = []
    for i_probe in range(N):
        pos = labels==labels[i_probe]
        pos[:i_probe+1] = False
        neg = pos.logical_not()
        neg[:i_probe+1] = False

        sim_2_gallery = similarity[i_probe, neg].tolist()
        sim_2_probe = similarity[i_probe, pos].tolist()

        non_match_similarity.extend(sim_2_gallery)
        match_similarity.extend(sim_2_probe)
    
    match_similarity.sort()
    non_match_similarity.sort()

    FA_count_1e_6 = int(1e-6 * len(non_match_similarity))
    t_1e_6 = 0.5 * (non_match_similarity[-FA_count_1e_6] + non_match_similarity[-FA_count_1e_6 - 1]) # thresholding at 1e-6 FAR
    TAR_1e_6 = (np.array(match_similarity) > t_1e_6).sum() / len(match_similarity)

    FA_count_1e_5 = int(1e-5 * len(non_match_similarity))
    t_1e_5 = 0.5 * (non_match_similarity[-FA_count_1e_5] + non_match_similarity[-FA_count_1e_5 - 1]) # thresholding at 1e-5 FAR
    TAR_1e_5 = (np.array(match_similarity) > t_1e_5).sum() / len(match_similarity)

    FA_count_1e_4 = int(1e-4 * len(non_match_similarity))
    t_1e_4 = 0.5 * (non_match_similarity[-FA_count_1e_4] + non_match_similarity[-FA_count_1e_4 - 1]) # thresholding at 1e-5 FAR
    TAR_1e_4 = (np.array(match_similarity) > t_1e_4).sum() / len(match_similarity)
    
    return TAR_1e_6, TAR_1e_5, TAR_1e_4

class process_train(object):
    def __init__(self, model, train_iter, eval_iter, LFW_dataloader, face_loss, fair_loss, logger, args):

        self.model = model
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.LFW_dataloader = LFW_dataloader
        self.face_loss = face_loss
        self.fair_loss = fair_loss

        self.emd_dim = args.emd_dim
        self.alpha = 1
        self.lmd1 = args.lmd1
        self.lmd2 = args.lmd2
        self.fair_loss_coef = args.fair_loss_coef
        self.fair_loss_alpha = args.fair_loss_alpha
        self.logger = logger
        self.device = args.device

        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.face_loss.parameters()), lr=args.lr, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.scheduler_steps, gamma=args.gamma)

    def train(self):

        self.model.train()
        self.face_loss.train()

        for idx, data in tqdm(enumerate(self.train_iter)):
            imgs = data[0]['imgs']
            labels = data[0]['labels']
            identities = labels[:,0].detach().type(torch.LongTensor).to(self.device)
            genders = labels[:,1].detach().type(torch.LongTensor).to(self.device)

            h = self.model(imgs)
            gender_vector = torch.ones([h.shape[0],1]).to(self.device)
            gender_vector[genders==0] = -1
            h = torch.cat([h,gender_vector], dim=1)
            face_loss = self.face_loss(h, identities)

            idx = [torch.where(identities==iden)[0][0].item() for iden in set(identities.tolist())]
            iden_unique = set(identities[idx].tolist())

            hn, hn_genders = self.face_loss.random_sample(1024, iden_unique)
            # hn, hn_genders = self.face_loss.random_sample(256, iden_unique) # originally 1024
            # hn = nn.functional.normalize(hn, p=2, dim=1, eps=1e-12)
            hn = nn.functional.normalize(hn[:,:-1], p=2, dim=1, eps=1e-12)
            fair_loss_1 = self.fair_loss(hn, hn_genders, self.fair_loss_alpha, self.fair_loss_coef)

            h2 = nn.functional.normalize(h[idx,:-1], p=2, dim=1, eps=1e-12)
            fair_loss_2  = self.fair_loss(h2, genders[idx], self.fair_loss_alpha, self.fair_loss_coef)
            loss = self.alpha / (self.alpha+self.lmd1+self.lmd2) * face_loss + self.lmd1/(self.alpha+self.lmd1+self.lmd2) * fair_loss_1 + self.lmd2/(self.alpha+self.lmd1+self.lmd2) * fair_loss_2

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        self.logger.log('train', 'loss', 'loss', loss.item())
        self.logger.log('train', 'loss', 'face_loss', face_loss.item())
        self.logger.log('whole', 'loss', 'fair_loss', fair_loss_1.item())
        self.logger.log('batch', 'loss', 'fair_loss', fair_loss_2.item())
        # self.logger.log('l2', 'pd2', 'pd2_whole', l2_1)
        # self.logger.log('l2', 'pd2', 'pd2_batch', l2_2)
        # self.logger.log('l2op', 'pd2', 'pd2_whole', l2op_1)
        # self.logger.log('l2op', 'pd2', 'pd2_batch', l2op_2)
        # self.logger.log('l2m', 'pd2', 'pd2_whole', l2m_1)
        # self.logger.log('l2m', 'pd2', 'pd2_batch', l2m_2)
        # self.logger.log('alpha', 'pd2', 'pd2_whole', alpha_1)
        # self.logger.log('alpha', 'pd2', 'pd2_batch', alpha_2)
        # self.logger.log('alphaop', 'pd2', 'pd2_whole', alphaop_1)
        # self.logger.log('alphaop', 'pd2', 'pd2_batch', alphaop_2)


    def test(self):

        self.model.eval()
        self.face_loss.eval()

        ##################################################################
        # validate on VGGFace2
        ##################################################################
        features = torch.empty([25000, self.emd_dim+1], device=self.device)
        idents = torch.empty([25000], device=self.device)

        idx_start = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.eval_iter):
                imgs = data[0]['imgs']
                labels = data[0]['labels']
                identities = labels[:,0].detach().type(torch.LongTensor).to(self.device)
                genders = labels[:,1].detach().type(torch.LongTensor).to(self.device)
                idx_end = idx_start + imgs.shape[0]

                h = self.model(imgs)
                gender_vector = torch.ones([h.shape[0],1]).to(self.device)
                gender_vector[genders==0] = -1
                h = torch.cat([h,gender_vector], dim=1)

                features[idx_start:idx_end, :] = h.detach()
                idents[idx_start:idx_end] = identities.detach()
                idx_start = idx_end
        
        normed_features = nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
        similarity = nn.functional.linear(normed_features, normed_features)
        R1_rate, R10_rate, R100_rate = eval_identification(similarity, idents)
        normed_features_2 = nn.functional.normalize(features[:,:-1], p=2, dim=1, eps=1e-12)
        similarity_2 = nn.functional.linear(normed_features_2, normed_features_2)
        R1_rate_2, R10_rate_2, R100_rate_2 = eval_identification(similarity_2, idents)
        # TAR_1e_6, TAR_1e_5, TAR_1e_4 = eval_verification(similarity, idents)

        self.logger.log('w/ gender', 'acc', 'rank_1_identification_rate', R1_rate)
        self.logger.log('w/ gender', 'acc', 'rank_10_identification_rate', R10_rate)
        self.logger.log('w/ gender', 'acc', 'rank_100_identification_rate', R100_rate)
        self.logger.log('w/o gender', 'acc', 'rank_1_identification_rate', R1_rate_2)
        self.logger.log('w/o gender', 'acc', 'rank_10_identification_rate', R10_rate_2)
        self.logger.log('w/o gender', 'acc', 'rank_100_identification_rate', R100_rate_2)
        # self.logger.log('test', 'acc', 'TAR_1e_6', TAR_1e_6)
        # self.logger.log('test', 'acc', 'TAR_1e_5', TAR_1e_5)
        # self.logger.log('test', 'acc', 'TAR_1e_4', TAR_1e_4)

        ##################################################################
        # validate on LFW
        ##################################################################
        similarity = torch.empty([6000])
        labels = torch.empty([6000], dtype=bool)

        idx_start = 0
        with torch.no_grad():
            for batch_idx, (img1, img2, label) in enumerate(self.LFW_dataloader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                idx_end = idx_start + img1.shape[0]

                feature1 = self.model(img1)
                feature2 = self.model(img2)

                normed_feature1 = nn.functional.normalize(feature1, p=2, dim=1, eps=1e-12)
                normed_feature2 = nn.functional.normalize(feature2, p=2, dim=1, eps=1e-12)

                if normed_feature1.shape[0] != normed_feature2.shape[0]:
                    raise ValueError('number of img1 not equal number of img2!')
                else:
                    similarity[idx_start:idx_end] = nn.functional.linear(normed_feature1, normed_feature2).diagonal()
                    labels[idx_start:idx_end] = label

                idx_start = idx_end
            
        LFW_tar, LFW_far = evaluate_lfw(np.array(similarity), np.array(labels))

        self.logger.log('test', 'acc', 'LFW_tar', LFW_tar)
        self.logger.log('test', 'acc', 'LFW_far', LFW_far)

