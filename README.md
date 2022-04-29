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

## Experiment on Adult
**Prepare Dataset**: Download the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) to `data/Adult`. This should produce the folder `data/Adult/Adult`, with files `adult.data`, `adult.names`, `adult.test`, and `old.adult.names`. Run the following data-preprocessing code. This should produce the file `Adult_fair.npz`.
```bash
cd data/Adult
python Adult_preprocess.py
cd ../..
```

**Train Model**: We use Visdom to monitor & visualize training, which requires additional setup. See [here](https://github.com/fossasia/visdom). Otherwise, run the following code.
```bash
python main_Adult.py @configs/Adult.txt
```

**Trained Models**: Five trained models with lengthscale 1, 2, and 2 square root of 2 can be found at `checkpoints/Adult`.

## Experiment on MPI3D
**Prepare Dataset**: Download the [MPI3D dataset real-world version](https://github.com/rr-learning/disentanglement_dataset) to `data/MPI3D`. This should produce the file `mpi3d_real.npz`. Run the following data-preprocessing code. This should produce the file `mpi3d_fair.npz`.
```bash
cd data/MPI3D
python MPI3D_preprocess.py
cd ../..
```

**Train Model**: We use Visdom to monitor & visualize training, which requires additional setup. See [here](https://github.com/fossasia/visdom). Otherwise, run the following code.
```bash
python main_MPI3D.py @configs/MPI3D.txt
```

**Trained Models**: All five trained ResNet-34 models---with which we report mean and std in paper---can be found at `checkpoints/MPI3D`.

## Experiment on VGGFace2
**Prepare Dataset**: Download and extract the [VGGFace2 dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2) to `data/VGGFace2_112_112/`. This should produce the folder `data/VGGFace2_112_112/dataset/` with subfolders `attributes`, `bb_landmark`, `test`, `train`, file `identity_meta.csv`, and other files. Run the following code to crop and align face images. This should produce new folders `data/VGGFace2_112_112/test` and `data/VGGFace2_112_112/train`.
```bash
cd data/VGGFace2_112_112
python face_crop_and_align.py
```

For fast IO, we pack the dataset to MXNet’s recordIO file by running the following code. 
```bash
# This will produce the file train.lst and test_500x50.lst
python create_lst.py 
# This will produce train.idx and train.rec
python im2rec.py train.lst train/ 
# This will produce test_500x50.idx and test_500x50.rec
python im2rec.py test_500x50.lst test/ 
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

**Train model**: We use Visdom to monitor & visualize training, which requires additional setup. See [here](https://github.com/fossasia/visdom). Otherwise, run the following code.
```bash 
python main_VGGFace2.py @configs/VGGFace2.txt
```

In learning gender-blind face representations with pretext loss ArcFace, we find it helpful to use MMD regularization twice each iteration, as line 133-141 from `lib/VGGFace2/process_train.py` shows. One is on the batch of training instances. Another is on 1024 randomly sampled entities' representation from ArcFace.

**Trained models**: All five trained Sphere20 models---with which we report mean and std in paper---can be found at `checkpoints/VGGFace2`. Two notes are in order: 1) face images need to be cropped and aligned in the same way the training data is preprocessed. 2) the model returns unnormalized face vectors, which does not necessarily anonymize gender. An additional l2 normalization is required before use.

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

def MMD2_rq_u(h, y, alpha=1, l2=1):
    """ finite-sample unbiased estimate for squared MMD with rational quadratic kernel
    Args:
        h (torch tensor, [N, d]): samples
        y (torch tensor, [N]): class, either 0 or 1
        alphas (list): a list of alphas, which we average over
    
    Returns:
        torch tensor, [1]: the finite-sample unbiased estimate
    """

    h_1 = h[[True if i==0 else False for i in y]]
    h_2 = h[[True if i==1 else False for i in y]]
    n_1 = h_1.shape[0]
    n_2 = h_2.shape[0]

    pd_12 = torch.cdist(h_1.unsqueeze(0),h_2.unsqueeze(0)).squeeze().reshape(-1)
    pd_11 = torch.pow(torch.nn.functional.pdist(h_1), 2)
    pd_22 = torch.pow(torch.nn.functional.pdist(h_2), 2)

    k_11 = torch.pow(1 + pd_11/(2 * alpha * l2), -alpha)
    k_22 = torch.pow(1 + pd_22/(2 * alpha * l2), -alpha)
    k_12 = torch.pow(1 + pd_12/(2 * alpha * l2), -alpha)
    out = ((k_11.sum() * 2) / (n_1*(n_1-1))
                    - 2 * k_12.sum() / (n_1 * n_2)
                    + (k_22.sum() * 2)/(n_2*(n_2-1)))
    return out
```

# Contact and Citation
Send any feedback to Xudong Shen (<xudong.shen@u.nus.edu>). Cite our work if you find our paper and/or the associated code helpful. : -)
```bibtex
@ARTICLE{2022_shen_fair_representation,
  author={Shen, Xudong and Wong, Yongkang and Kankanhalli, Mohan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Fair Representation: Guaranteeing Approximate Multiple Group Fairness for Unknown Tasks}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2022.3148905}}
```
