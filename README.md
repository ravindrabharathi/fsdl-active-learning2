# fsdl-active-learning

Comparing different active learning strategies for image classification (FSDL course 2021 capstone project)

- [fsdl-active-learning](#fsdl-active-learning)
  - [Introduction](#introduction)
  - [Relevant Changes Compared to Lab Template](#relevant-changes-compared-to-lab-template)
    - [DroughtWatch Data Set](#droughtwatch-data-set)
    - [ResNet Image Classifier](#resnet-image-classifier)
    - [modAL Active Learning Experiment Running Framework](#modal-active-learning-experiment-running-framework)
      - [modAL Sampling Strategy Extensions](#modal-sampling-strategy-extensions)
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

## Quickstart

### Local

```bash
git pull [repo-url] # clone from git
cd [folder]

make conda-update # creates a conda env with the base packages
conda activate fsdl-active-learning-2021 # activates the conda env
make pip-tools # installs required pip packages inside the conda env

# regular experiment, training a drought watch classifier via pytorch lightning
python training/run_experiment.py --max_epochs=1 --num_workers=4 --data_class=DroughtWatch --model_class=ResnetClassifier
```

### Google Colab

Refer to the example notebook under [notebooks folder](./notebooks).
