from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import numpy as np
import random

class ArcFace(nn.Module):
    def __init__(self, emdsize: int ,class_num: int, s: float,  m:float, device: str, lst_file: str) -> None:
        super(ArcFace, self).__init__()
        self.class_num = class_num
        self.emdsize = emdsize
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.empty([self.class_num, self.emdsize], device=device))
        nn.init.xavier_uniform_(self.weight)

        self.CEL = torch.nn.CrossEntropyLoss()
        self.device = device

        self.genders = self.read_genders(lst_file)
        self.gender_vector = torch.ones([self.weight.shape[0],1]).to(self.device)
        self.gender_vector[self.genders==0] = -1
        self.idx = set(range(0,class_num))
    
    def read_genders(self, lst_file):
        genders = -1 * np.ones(self.class_num)
        with open(lst_file, 'r') as f:
            for line in f:
                _, iden, gender, _ = line.strip().split('\t')
                iden = int(iden)
                gender = int(gender)
                genders[iden] = gender
            f.close()
        if (genders == -1).any():
            raise ValueError('Arcface loss initializing failed.')
        else:
            return genders
    
    def random_sample(self, num, iden_unique):
        if num - len(iden_unique) > 0:
            idx = random.sample(self.idx - iden_unique, num - len(iden_unique))
        else:
            idx = []
        idx = idx + list(iden_unique)
        return self.weight[idx], self.genders[idx]

    def forward(self, input: Tensor, label: Tensor) -> Tensor:

        normed_input = nn.functional.normalize(input, p=2, dim=1, eps=1e-12)
        # normed_weight = nn.functional.normalize(torch.cat([self.weight,self.gender_vector], dim=1), p=2, dim=1, eps=1e-12)
        normed_weight = nn.functional.normalize(self.weight, p=2, dim=1, eps=1e-12)

        cosine = F.linear(normed_input, normed_weight).clamp(-1, 1)

        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        cosine_t = cosine[one_hot]

        sine_t = torch.sqrt(1.0 - torch.pow(cosine_t, 2))
        phi_t = cosine_t * self.cos_m - sine_t * self.sin_m

        phi = torch.where(cosine_t > self.th, phi_t, cosine_t - self.mm)

        cosine[one_hot] = phi

        return self.CEL(self.s * cosine, label)
