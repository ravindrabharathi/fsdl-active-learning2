# fsdl-active-learning

Comparing different active learning strategies for image classification (FSDL course 2021 capstone project)

- [fsdl-active-learning](#fsdl-active-learning)
  - [Introduction](#introduction)
  - [Relevant Changes Compared to Lab Template](#relevant-changes-compared-to-lab-template)
  - [Quickstart](#quickstart)
    - [Local](#local)
    - [Google Colab](#google-colab)
  - [Documentation](#documentation)

## Introduction

This repository builds upon the template of **lab 08** of the [Full Stack Deep Learning Spring 2021 labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and extends it with new datasets, models and a full active learning strategy experiment framework.

It was implemented as capstone project for the Spring course 2021 by [Stefan Josef](https://www.linkedin.com/in/stefan-j-7a5a6b120/), [Matthias Pfenninger](https://www.linkedin.com/in/matthiaspfenninger/) and [Ravindra Bharati](https://www.linkedin.com/in/sravindrabharathi/).

## Relevant Changes Compared to Lab Template

**Datasets**: [DroughtWatch](https://github.com/wandb/droughtwatch) and [MNIST](https://en.wikipedia.org/wiki/MNIST_database)

**Models**: PyTorch's [ResNet50](https://pytorch.org/hub/pytorch_vision_resnet/) extended with own input and output layers for both datasets mentioned above

**Active learning experiment frameworks**: Self-developed extension of the course's BaseDataModule that builds on top of a PyTorch's LightningDataModule (see [training/run_experiment.py](./training/run_experiment.py)) and separate experiment routine that uses the [modAL library](https://github.com/modAL-python/modAL) (see [training/run_modal_experiment.py](./training/run_modal_experiment.py))

**Active learning sampling strategies**: The following sampling strategies are available:

- Uncertainty Sampling
  - least_confidence
  - margin
  - ratio
  - entropy
  - least_confidence_pt
  - margin_ptmargin_pt
  - ratio_pt
  - entropy_pt
- Bayesian Uncertainty Sampling
  - bald
  - max_entropy
  - least_confidence_mc
  - margin_mc
  - ratio_mc
  - entropy_mc
- Diversity Sampling
  - mb_outliers_mean
  - mb_outliers_max
  - mb_clustering
  - mb_outliers_glosh
- Mixed Sampling
  - mb_outliers_mean_least_confidence
  - mb_outliers_mean_entropy
- Other Advanced Strategies
  - active_transfer_learning
  - dal
- Baseline
  - random

## Quickstart

### Local

```bash
git pull https://github.com/ravindrabharathi/fsdl-active-learning2 # clone from git
cd fsdl-active-learning2

make conda-update # creates a conda env with the base packages
conda activate fsdl-active-learning-2021 # activates the conda env
make pip-tools # installs required pip packages inside the conda env

# active learning experiment with DroughtWatch
python training/run_experiment.py \
  --sampling_method=active_transfer_learning \
  --data_class=DroughtWatch \
  --model_class=ResnetClassifier \
  --n_train_images=1000 \
  --al_samples_per_iter=500 \
  --al_iter=20 \
  --max_epochs=20 \
  --pretrained=True \
  --binary \
  --rgb \
  --lr=3e-4 \
  --gpus=1 \
  --wandb

# active learning experiment with MNIST
python training/run_experiment.py \
  --data_class=MNIST \
  --model_class=MNISTResnetClassifier \
  --gpus=1 \
  --wandb

# active learning experiment via modAL framework
python training/run_modal_experiment.py \
  --data_class=DroughtWatch \
  --model_class=ResnetClassifier \
  --al_query_strategy=margin_sampling
  --gpus=1 \
  --wandb

```

### Google Colab

```bash
# clone project from github
!git clone https://github.com/ravindrabharathi/fsdl-active-learning2
%cd fsdl-active-learning2
```

```bash
# install necessary packages and add library directory to your pythonpath
!pip3 install boltons wandb pytorch_lightning==1.2.8 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
!pip3 install modAL tensorflow skorch hdbscan

%env PYTHONPATH=.:$PYTHONPATH
```

```bash
# initialize w&b with your personal info
!wandb login your_wandb_key
!wandb init --project your_wandb_project --entity your_wandb_entity #
```

```bash
# start experimenting
!python training/run_experiment.py \
  --data_class=MNIST \
  --model_class=MNISTResnetClassifier \
  --gpus=1 \
  --wandb
```

For more examples refer to the notebooks in the [notebooks folder](./notebooks).

## Documentation

For more details please refer to the [documentation and detailed project report](./docs).

## Contributing

We are happy if you want to contribute. Contact us on LinkedIn (see links above) if you want to discuss anything or open an issue here in the repository.
