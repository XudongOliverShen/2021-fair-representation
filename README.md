# Introduction

This repository contains the code associated with the upcoming paper in TPAMI titled "[Fair Representation: Guaranteeing Approximate Multiple Group Fairness for Unknown Tasks](https://arxiv.org/abs/2109.00545)".

# Fairness Guarantees

In [paper](https://arxiv.org/abs/2109.00545), we prove that the downstream unknown prediction task's fairness can be approximately guaranteed w.r.t. seven fairness notions simultaneously, if the predictions are made using fair representations. These fairness guarantees (tight upper bounds) can be found by solving linear programs parameterized by the representation's fairness coefficient $\alpha$, discriminativeness coefficient $\beta$, and the population base rates $a$, $b$, $r$. We implement this linear program in `main_lp.py`.

We use the commercial software Gurobi as the solver. An unlimited [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) can be easily obtained.  The required Python package is follows:

```txt
gurobipy==9.5.0
```

# Learning Fair Representation

We propose to learn both fair and discriminative representations using pretext loss, which self-supervises the representation to summarize all semantics from the data, and Maximum Mean Discrepancy, which is used as a fair regularization.

## Prerequisites
- Recommended environment: Ubuntu 18.04.5 LTS with CUDA 11.3.
- Required Python packages listed in `requirements.txt`.

## Experiment on VGGFace2
1. **Prepare Dataset**: Download and extract the [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2) to `data/VGGFace2_112_112/`. This should produce the folder `data/VGGFace2_112_112/dataset/` with subfolders `attributes`, `bb_landmark`, `test`, `train`, file `identity_meta.csv`, and other files. Run the following code to crop and align face images. This should produce new folders `data/VGGFace2_112_112/test` and `data/VGGFace2_112_112/train`.
```bash
cd data/VGGFace2_112_112
python face_crop_and_align.py
```

For fast IO, we pack the dataset to MXNet’s recordIO file by running the following code. 
```bash
# This will produce the file train.lst and test_500x50.lst
python create_lst.py 
# This will produce train.idx and train.rec
python ../im2rec.py train.lst train/ 
# This will produce test_500x50.idx and test_500x50.rec
python ../im2rec.py test_500x50.lst test/ 
cd ../..
```

Download and extract the [LFW dataset](http://vis-www.cs.umass.edu/lfw/) to `data/LFW_112_112`. This should produce the folder `data/LFW_112_112/lfw`. run the following code.
```bash
cd data/LFW_112_112
# This will produce the folder test/
python face_crop_and_align.py
# This will produce the file test.lst
python create_lst.py
# This will produce the file test.idx and test.rec
python ../im2rec.py test.lst test/
cd ../..
```

2. **Train model**: We use Visdom to monitor & visualize training, which requires additional setup. See [here](https://github.com/fossasia/visdom). Otherwise, run the following code.
```bash 
python main_VGGFace2.py @configs/VGGFace2.txt
```

In learning gender-blind face representations with pretext loss ArcFace, we find it helpful to use MMD regularization twice each iteration, as line 133-141 from `lib/VGGFace2/process_train.py` shows. One is on the batch of training instances. Another is on 1024 randomly sampled entities' representation from ArcFace.

3. **Trained models**: All five trained Sphere20 neural models---with which we report mean and std in paper---can be found at `checkpoints/VGGFace2`. Two notes are in order: 1) face images need to be cropped and aligned in the same way the training data is preprocessed. 2) the model returns unnormalized face vectors, which does not necessarily anonymize gender. An additional l2 normalization is required before use.

## Implementation of Maximum Mean Discrepancy as fair regularizer
To some who may find it useful, our implementation of MMD with rational quadratic kernel as a fair regularizer is follows.
```python
import torch
import math
import numpy as np
import statistics
from scipy import optimize

def pdist_squared(sample_1, sample_2):
    """ squared pair-wise distance

    Args:
        samples_1 (torch tensor, [n_1, d]): the first set of samples
        samples_2 (torch tensor, [n_2, d]): the second set of samples
    Returns:
        a matrix of pair-wise distance (torch tensor, [n_1, n_2])
    """

    n_1, n_2 = sample_1.size(0), sample_2.size(0)

    sample_1 = sample_1.unsqueeze(1).repeat(1,n_2,1)
    sample_2 = sample_2.unsqueeze(0).repeat(n_1,1,1)

    distances_squared = torch.pow(sample_1 - sample_2, 2).sum(dim=-1)

    return distances_squared

def gram_with_RQ_kernel(sample_1, sample_2, alpha, l):
    """ calculate the gram matrix for RQ kernel between two sets of samples

    k^{rq}_{\alpha} (x, y) = 
        (1 + \frac{\|x-y\|^2}{2\alpha})^{-\alpha}
    
    Args:
        samples_1 (torch tensor, [n_1, d]): the first set of samples
        samples_2 (torch tensor, [n_2, d]): the second set of samples
        alpha: the parameter in RQ kernel
    
    Returns:
        the gram matrix (torch tensor, [n_1, n_2])
    """

    distances_squared = pdist_squared(sample_1, sample_2)

    return torch.pow(1 + distances_squared/(2*alpha*l**2), -alpha)

def MMD2_rq_b(h, y, alpha, l):
    """ finite-sample biased estimate for squared MMD with rational quadratic kernel

    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]

    out = (gram_with_RQ_kernel(h_1, h_1, alpha, l).mean() 
                - 2 * gram_with_RQ_kernel(h_1, h_2, alpha, l).mean() 
                + gram_with_RQ_kernel(h_2, h_2, alpha, l).mean())
    return out
```

# Citation

Cite our work, or not : -), if you find our paper and/or the associated code helpful.
```bibtex
@misc{shen2021fair,
      title={Fair Representation: Guaranteeing Approximate Multiple Group Fairness for Unknown Tasks}, 
      author={Xudong Shen and Yongkang Wong and Mohan Kankanhalli},
      year={2021},
      eprint={2109.00545},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```