# MWLDAIML

This repository contains the code and datasets for **Metric Learning based Weighted Linear Discriminant Analysis for Imbalanced Multi-Label Classification**, including:

- Original model  
- Semi-supervised model  
- Experiments with feature noise

---

## 1. System Requirements

The code was developed and tested on:

- **CPU:** 13th Gen Intel(R) Core(TM) i5-13600KF 3.50GHz  
- **Memory:** 32 GB RAM  
- **GPU:** NVIDIA GeForce RTX 4070 with 27.9 GB memory  
- **OS:** 64-bit Windows 10  
- **Python version:** >= 3.8  

> For GPU-accelerated computations, ensure that PyTorch is installed with CUDA support.

---

## 2. Usage

1. Clone the repository:

```bash
git clone https://github.com/hjttnt/code-of-MWLDAIML.git
cd code-of-MWLDAIML
```
The scripts expect datasets to be located at specific paths in test.py, test_semi.py and test_noisy.py. Ensure that your folder structure matches the paths in the scripts.

---

2. Usage
2.1 Original Model

```bash
python test.py

2.2 Semi-Supervised Model

```bash
python test_semi.py

2.3 Noisy Dataset Experiments

```bash
python test_noisy.py

Results and logs will be saved according to the configuration in each script.
