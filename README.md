# fsdl-active-learning

Comparing different active learning strategies for image classification (FSDL course 2021 capstone project)

- [fsdl-active-learning](#fsdl-active-learning)
  - [Introduction](#introduction)
  - [Relevant Changes Compared to Lab Template](#relevant-changes-compared-to-lab-template)
    - [DroughtWatch Data Set](#droughtwatch-data-set)
    - [ResNet Image Classifier](#resnet-image-classifier)
    - [Main Active Learning Experiment Running Framework](#main-active-learning-experiment-running-framework)
      - [Uncertainty Sampling](#uncertainty-sampling)
        - [least_confidence](#least_confidence)
        - [margin](#margin)
        - [ratio](#ratio)
        - [entropy](#entropy)
        - [least_confidence_pt](#least_confidence_pt)
        - [margin_pt](#margin_pt)
        - [ratio_pt](#ratio_pt)
        - [entropy_pt](#entropy_pt)
      - [Bayesian Uncertainty Sampling](#bayesian-uncertainty-sampling)
        - [bald](#bald)
        - [max_entropy](#max_entropy)
        - [least_confidence_mc](#least_confidence_mc)
        - [margin_mc](#margin_mc)
        - [ratio_mc](#ratio_mc)
        - [entropy_mc](#entropy_mc)
      - [Diversity Sampling](#diversity-sampling)
        - [mb_outliers_mean](#mb_outliers_mean)
        - [mb_outliers_max](#mb_outliers_max)
        - [mb_clustering](#mb_clustering)
        - [mb_outliers_glosh](#mb_outliers_glosh)
      - [Advanced Sampling Techniques](#advanced-sampling-techniques)
        - [mb_outliers_mean_least_confidence](#mb_outliers_mean_least_confidence)
        - [mb_outliers_mean_entropy](#mb_outliers_mean_entropy)
        - [active_transfer_learning](#active_transfer_learning)
        - [dal](#dal)
      - [Baseline](#baseline)
        - [random](#random)
    - [modAL Active Learning Experiment Running Framework](#modal-active-learning-experiment-running-framework)
      - [Bayesian Uncertainty Sampling](#bayesian-uncertainty-sampling-1)
        - [bald](#bald-1)
        - [max_entropy](#max_entropy-1)
      - [Diversity Sampling](#diversity-sampling)
        - [outlier](#outlier)
        - [cluster_outlier_combined](#cluster_outlier_combined)
      - [Baseline](#baseline-1)
        - [random](#random-1)
  - [Quickstart](#quickstart)
    - [Local](#local)
    - [Google Colab](#google-colab)

## Introduction

This repository builds upon the template of **lab 08** of the [Full Stack Deep Learning Spring 2021 labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and extends it with a new dataset, model and active learning strategies.

## Relevant Changes Compared to Lab Template

### DroughtWatch Data Set

[text_recognizer/data/droughtwatch.py](./text_recognizer/data/droughtwatch.py): Downloads data from the [W&B Drought Prediction Benchmark](https://github.com/wandb/droughtwatch) and converts it to HDF5 format which can be used by PyTorch for training and inference.

### ResNet Image Classifier

[text_recognizer/models/resnet_classifier.py](./text_recognizer/models/resnet_classifier.py): Implements a PyTorch ResNet model for image classification, with the following adaptions compared to the regular model:

- preprocessing steps (image resizing and normalization)
- class outputs (4 instead of 1000)
- optional dropout layer at the end

The model can be used for transfer learning on the drought prediction data.

### Main Active Learning Experiment Running Framework

[training/run_experiment.py](./training/run_experiment.py): Script to run experiments for model training with different active learning strategies which are implemented in the separate submodule [active_learning/sampling/al_sampler.py](./active_learning/sampling/al_sampler.py).

#### Uncertainty Sampling

##### least_confidence

##### margin

##### ratio

##### entropy

##### least_confidence_pt

##### margin_pt

##### ratio_pt

##### entropy_pt

#### Bayesian Uncertainty Sampling

##### bald

##### max_entropy

##### least_confidence_mc

##### margin_mc

##### ratio_mc

##### entropy_mc

#### Diversity Sampling

##### mb_outliers_mean

##### mb_outliers_max

##### mb_clustering

##### mb_outliers_glosh

#### Advanced Sampling Techniques

##### mb_outliers_mean_least_confidence

##### mb_outliers_mean_entropy

##### active_transfer_learning

##### dal

#### Baseline

##### random

### modAL Active Learning Experiment Running Framework

[training/run_modAL_experiment.py](./training/run_modal_experiment.py): Script to run experiments for model training with different active learning strategies which are implemented via the [modAL library](https://github.com/modAL-python/modAL). The modAL extensions are bundled under the submodule [active_learning/sampling/modal_extensions.py](./active_learning/sampling/modal_extensions.py).

Note that the strategies `bald` and `max_entropy` only make sense when there is a `dropout` layer in the network.

#### Bayesian Uncertainty Sampling

##### bald

Active learning sampling technique that maximizes the information gain via maximising mutual information between predictions and model posterior (Bayesian Active Learning by Disagreement - BALD) as depicted in the papers [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf) and [Bayesian Active Learning for Classification and Preference Learning](https://arxiv.org/pdf/1112.5745.pdf).

##### max_entropy

Active learning sampling technique that maximizes the predictive entropy based on the paper [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf).

#### Diversity Sampling

##### outlier

Self-developed outlier sampling technique:

1. Translates instances from the pool to features by tapping into an intermediate layer of the adapted ResNet model
2. Performs clustering via HDBSCAN and assigns an outlier score to each instance via GLOSH
3. Samples instances with highest outlier scores

##### cluster_outlier_combined

Self-developed diversity sampling technique that combines clustering and outliers:

1. Translates instances from the pool to features by tapping into an intermediate layer of the adapted ResNet model
2. Performs clustering via HDBSCAN and additionally assigns an outlier score to each instance via GLOSH
3. Samples instances evenly from all clusters, and takes both outlier and non-outlier points by considering the outlier score

#### Baseline

##### random

Baseline active learning sampling technique that takes random instances from available pool.

## Quickstart

### Local

```bash
git pull [repo-url] # clone from git
cd [folder]

make conda-update # creates a conda env with the base packages
conda activate fsdl-active-learning-2021 # activates the conda env
make pip-tools # installs required pip packages inside the conda env

# active learning experiment
python training/run_experiment.py --sampling_method=max_entropy --data_class=DroughtWatch --model_class=ResnetClassifier --batch_size=64 --gpus=1 --max_epochs=20

# active learning experiment with modAL
python training/run_modaL_experiment.py --al_epochs_init=10 --al_epochs_incr=10 --al_n_iter=20 --al_samples_per_iter=2000 --al_incr_onlynew=False --al_query_strategy=bald --data_class=DroughtWatch --model_class=ResnetClassifier --batch_size=64 --n_train_images=20000 --n_validation_images=10778  --pretrained=True --wandb
```

### Google Colab

Refer to the example notebook under [notebooks folder](./notebooks).
