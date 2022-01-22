# Introduction

This repository contains the code associated with the upcoming paper in TPAMI titled "[Fair Representation: Guaranteeing Approximate Multiple Group Fairness for Unknown Tasks](https://arxiv.org/abs/2109.00545)".

# Fairness Guarantees

In [paper](https://arxiv.org/abs/2109.00545), we prove that the downstream prediction task's fairness can be approximately guaranteed w.r.t. multiple fairness notions simultaneously, if the predictions are made using fair representations. These fairness guarantees (upper bounds) can be found by solving linear programs parameterized by the representation's fairness coefficient $\alpha$, discriminativeness coefficient $\beta$, and the population base rates $a$, $b$, $r$. We implement this linear program in `main_lp.py`.

We use the commercial software Gurobi as the solver. An unlimited [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) can be easily obtained.  The required Python package is follows:

```txt
gurobipy==9.5.0
```

# Learning Fair Representation

We propose to learn both fair and discriminative representations using pretext loss, which self-supervises the representation to summarize all semantics from the data, and Maximum Mean Discrepancy, which is used as a fair regularization.

# Citation

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