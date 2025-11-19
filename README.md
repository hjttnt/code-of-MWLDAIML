# MWLDAIML

This repository contains the code and datasets for [Metric learning based weighted linear discriminant analysis for imbalanced multi-label classification], including the original model, a semi-supervised variant, and experiments with feature noise.

## System Requirements

The code was developed and tested on the following hardware and software environment:

- **CPU:** 13th Gen Intel(R) Core(TM) i5-13600KF 3.50GHz  
- **Memory:** 32 GB RAM  
- **GPU:** NVIDIA GeForce RTX 4070 with 27.9 GB memory  
- **OS:** 64-bit Windows 10  
- **Python version:** >= 3.8 

> Ensure that your system meets these requirements for optimal performance, especially for GPU-accelerated computations.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/hjttnt/code-of-MWLDAIML.git
cd code-of-MWLDAIML

## Dataset and Code Path Configuration

The scripts expect the datasets to be located at specific paths in test.py, test_semi.py and test_noisy.py. Make sure your folder structure matches the paths used in the scripts.

---

## Usage

### Original Model

```bash
python test.py


### Semi-Supervised Model

```bash
python test_semi.py

### Noisy Dataset Experiments

```bash
python test_noisy.py
