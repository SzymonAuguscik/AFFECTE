# AFFECTE - Atrial Fibrillation Finder from Electrocardiogram with Convolution and Transformer Encoder

*AFFECTE* is a Python3.10 package to build a Transformer-based neural network to detect atrial fibrillation using ECG signal.
It was inspired by [Constrained transformer network for ECG signal processing and arrhythmia classification](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01546-2). 
The models were trained on [MIT-BIH Long Term AF Database](https://physionet.org/content/ltafdb/1.0.0/).

Table of contents:
- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Performance](#performance)
- [Future work](#future-work)

## Features

The *affecte.py* script contains the essential features from *AFFECTE* package, including:
- dataset preparation
- hyperparameters setting (especially related with Transformer network)
- automatic architecture choice
- learning process execution
- model evaluation
- results visualization.

Additionally, it provides examples of learning support techniques, e.g. data augmentation or cross validation.

## Architecture

The *AFFECTE* architecture relies on user's choices. The primary (full) version consists of three subnetworks: convolution neural network (CNN), Transformer and feed forward classifier (FFC) and is implemented as *AtrialFibrillationDetector* class.
Depending on the user's preferences, CNN and/or Transformer can be skipped.
The software will automatically align the necessary dimension along the main network.

## Setup

Clone repository and go inside:
```bash
git clone https://github.com/SzymonAuguscik/AFFECTE
cd AFFECTE
```

Create virtual environment for *AFFECTE* and activate it:
```bash
python3.10 -m venv affecte_venv
source affecte_venv/bin/activate
```

Install required libraries:
```bash
venv/bin/pip3 install -r requirements.txt
```

Create data directory with necessary subfolders:
```bash
mkdir -p data/archives data/datasets data/long-term-af-database-1.0.0
```

Download MIT-BIT Long Term AF Database and unzip it:
```bash
wget -P data/archives https://physionet.org/static/published-projects/ltafdb/long-term-af-database-1.0.0.zip
unzip data/archives/long-term-af-database-1.0.0.zip -d data/long-term-af-database-1.0.0/
```

Verify setup by running exemplary script:
```bash
affecte_venv/bin/python3.10 affecte.py
```

## Usage

The *affecte.py* script can be used as a demo or to repeat author's experiments.
Usage:
```bash
affecte [-h]
        [-c CHANNEL [CHANNEL ...]]
        [-s [1-60]]
        [-e [1-500]]
        [-l LEARNING_RATE]
        [-b [1-2048]]
        [-t [1-2048]]
        [-f [1-2048]]
        [-a [1-128]]
        [-n [1-64]]
        [-d [1-1000000]]
        [-v [0-4]]
        [--use_cnn]
        [--use_transformer]
```
- **-h, --help**  
show this help message and exit
- **-c CHANNEL [CHANNEL ...], --channels CHANNEL [CHANNEL ...]**  
ECG channels to be used in the training (default: [0, 1])
- **-s [1-60], --seconds [1-60]**  
length of the windows that the ECG signal will be split (default: 10)
- **-e [1-500], --epochs [1-500]**  
training iterations (default: 50)
- **-l LEARNING_RATE, --learning_rate LEARNING_RATE**  
learning rate to be used in the optimizer (default: 0.0001)
- **-b [1-2048], --batch_size [1-2048]**  
the data samples used per epoch (default: 128)
- **-t [1-2048], --transformer_dimension [1-2048]**  
d_model for Transformer (default: 128)
- **-f [1-2048], --transformer_hidden_dimension [1-2048]**  
dimension of feed forward layers in Transformer (default: 256)
- **-a [1-128], --transformer_heads [1-128]**  
number of attention heads per encoder in Transformer (default: 16)
- **-n [1-64], --transformer_encoder_layers [1-64]**  
number of encoding layers in Transformer (default: 8)
- **-d [1-1000000], --dataset_custom_size [1-1000000]**  
set how many samples should be used from dataset; use all samples if not set (default: None)
- **-v [0-4], --validation_step [0-4]**  
choose which cross validation fold should be used as test set (default: 0)
- **--use_cnn**  
if set, use CNN layer in the main model (default: False)
- **--use_transformer**  
if set, use Transformer layer in the main model (default: False)

## Best results

So far the best mean accuracy score achieved by a model during five-fold cross validation was ***90.8% &plusmn; 1.0%***.  
The most accurate model and learning configurations:
- Transformer dimension: 32
- Transformer hidden dimension: 256
- Transformer encoders: 8
- Transformer heads: 32
- epochs: 50
- learning rate: 0.001
- other parameters: default

## Future work

- [x] Add docstrings
- [x] Add UTs
- [ ] Split classes into smaller ones
