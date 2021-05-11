"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from sklearn.model_selection import train_test_split

from active_learning import lit_models
from active_learning.data.util import BaseDataset
from active_learning.lit_models.base import MaxAccuracyLogger

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)

DEBUG_OUTPUT = True
MC_ITERATIONS = 10

BASIC_SAMPLING_METHODS = ["random", "least_confidence", "margin", "ratio", "entropy", "least_confidence_pt", "margin_pt", "ratio_pt", "entropy_pt"]
MC_SAMPLING_METHODS = ["bald", "max_entropy", "least_confidence_mc", "margin_mc", "ratio_mc", "entropy_mc"]
MB_SAMPLING_METHODS = ["mb_outliers_mean", "mb_outliers_max", "mb_outliers_mean_least_confidence", "mb_outliers_mean_entropy", "mb_outliers_glosh", "mb_clustering"]
ATL_SAMPLING_METHODS = ["active_transfer_learning", "DAL"]


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'active_learning.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="DroughtWatch")
    parser.add_argument("--model_class", type=str, default="ResnetClassifier")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="No. of epochs without improvement before stopping early")

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"active_learning.data.{temp_args.data_class}")
    model_class = _import_class(f"active_learning.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    # Active learning specific arguments
    parser.add_argument("--sampling_method", type=str, choices=BASIC_SAMPLING_METHODS+MC_SAMPLING_METHODS+MB_SAMPLING_METHODS+ATL_SAMPLING_METHODS, default="random", help="Active learning sampling strategy")
    parser.add_argument("--al_iter", type=int, default=-1, help="No. of active learning iterations (-1 to iterate until pool is exhausted)")
    parser.add_argument("--al_samples_per_iter", type=int, default=2000, help="No. of samples to query per active learning iteration")
    parser.add_argument("--al_continue_training", action='store_true', help="Whether to continue training after sampling from pool (instead of training from scratch each time")

    parser.add_argument("--help", "-h", action="help")
    return parser


def _initialize_trainer(model_class, lit_model_class, data, args, logger, al_iteration):
    
    print(f"Initializing model for active learning iteration {al_iteration}")

    # initialize model
    model = model_class(data_config=data.config(), args=args)
    logger.watch(model)
    
    # initialize lit_model
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    # initialize callbacks
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=args.early_stopping_patience)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    max_accuracy_callback = MaxAccuracyLogger()
    callbacks = [early_stopping_callback, model_checkpoint_callback, max_accuracy_callback]

    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    # for active transfer learning methods disactivate sanity check
    if args.sampling_method in ATL_SAMPLING_METHODS:
        trainer.num_sanity_val_steps = 0

    return trainer, lit_model


def _prepare_dataloaders(new_data, new_labels, data, args, val_split=0.2):

    # shuffle dataset
    randomize = np.arange(len(new_data))
    np.random.shuffle(randomize)
    new_data = new_data[randomize]
    new_labels = new_labels[randomize]

    # create training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_labels, test_size=val_split, random_state=42)

    # create datasets
    new_train_ds = BaseDataset(X_train, y_train, transform=data.transform)
    new_val_ds = BaseDataset(X_test, y_test, transform=data.transform)

    # create dataloaders
    new_train_dataloader = torch.utils.data.DataLoader(new_train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    new_val_dataloader = torch.utils.data.DataLoader(new_val_ds, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    return new_train_dataloader, new_val_dataloader


def _finetune_and_sample(lit_model, data, new_train_dataloader, new_val_dataloader, sample_size, args):

    print(f"\nStarting fine-tuning for {args.sampling_method}")

    # create new classification head
    new_head_part_2 = nn.Sequential(
        nn.BatchNorm1d(lit_model.model.head_part_1[3].out_features), # adding batchnorm
        nn.ReLU(), # additional nonlinearity
        nn.Dropout(lit_model.model.head_part_1[2].p), # additional dropout layer
        nn.Linear(lit_model.model.head_part_1[3].out_features, 2) # same fc layer as we had before
    )

    # replace lit model head
    lit_model.model.head_part_2 = new_head_part_2

    # freeze Resnet backbone
    for p in lit_model.model.resnet.parameters():
        p.requires_grad = False

    # turn off logging
    lit_model.logging = False

    # initialize trainer
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2)
    trainer = pl.Trainer(gpus=args.gpus, callbacks=[early_stopping_callback] progress_bar_refresh_rate=30, max_epochs=10, num_sanity_val_steps=0)
    trainer.tune(lit_model, train_dataloader=new_train_dataloader, val_dataloaders=new_val_dataloader)

    # fit trainer on new data
    trainer.fit(lit_model, train_dataloader=new_train_dataloader, val_dataloaders=new_val_dataloader)

    # run inference on unlabelled pool
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool_dl = data.unlabelled_dataloader()
    all_preds = torch.Tensor().to(device)
    lit_model.to(device)
    lit_model.eval()
    with torch.no_grad():
        for x,_ in pool_dl:
            logits = lit_model(x.to(device))
            preds = nn.functional.softmax(logits, dim=-1)
            all_preds = torch.cat([all_preds, preds])
    
    # sample indices from prediction scores
    _, idxs = torch.topk(all_preds[:,0], sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"active_learning.data.{args.data_class}")
    model_class = _import_class(f"active_learning.models.{args.model_class}")

    data = data_class(args)

    sampling_method = args.sampling_method
    if not sampling_method in ATL_SAMPLING_METHODS:
        sampling_class = _import_class(f"active_learning.sampling.al_sampler.{sampling_method}")

    lit_model_class = lit_models.BaseLitModel

    # get info about scenario for logging
    if args.binary:
        class_scenario = "_binary"
    else:
        class_scenario = "_multi-class"
    if args.rgb:
        channel_scenario = "_rgb"
    else:
        channel_scenario = "_all-channels"

    # --wandb args parameter ignored for now, always using wandb
    # specify project & entity outside of code, example: wandb init --project fdsl-active-learning --entity fsdl_active_learners
    project_name = "fsdl-active-learning_" + sampling_method + class_scenario + channel_scenario
    logger = pl.loggers.WandbLogger(name=project_name, job_type="train") 
    logger.log_hyperparams(vars(args))

    args.weights_summary = None  # set to "full" to print model layer summary
    unlabelled_data_size = data.get_ds_length(ds_name="unlabelled")

    al_iteration = 0

    trainer, lit_model = _initialize_trainer(model_class, lit_model_class, data, args, logger, al_iteration)
    
    while unlabelled_data_size > 0 and (args.al_iter < 0 or al_iteration < args.al_iter):

        # fit model on current data
        trainer.fit(lit_model, datamodule=data)

        # log best accuracies of this iteration to wandb
        wandb.log({
            "train_size": data.get_ds_length(ds_name="train"),
            "train_acc_best": trainer.callback_metrics.get("train_acc_max"),
            "val_acc_best": trainer.callback_metrics.get("val_acc_max"),
            "train_f1_best": trainer.callback_metrics.get("train_f1_max"),
            "val_f1_best": trainer.callback_metrics.get("val_f1_max"),
        })

        # if pool is large enough, take 'al_samples_per_iter' new samples - otherwise take all remaining ones
        if unlabelled_data_size > args.al_samples_per_iter:
            sample_size = args.al_samples_per_iter
        else:
            sample_size = unlabelled_data_size

        print("Total Unlabelled Pool Size ", unlabelled_data_size)
        print("Query Sample size ", sample_size)

        if sampling_method in MC_SAMPLING_METHODS:

            # predict multiple times with model in 'training' mode (dropout activated)
            probabilities = data.get_pool_probabilities(lit_model, T=MC_ITERATIONS)

            new_indices = sampling_class(probabilities, sample_size)

        elif sampling_method in MB_SAMPLING_METHODS:

            # get activations from intermediate layers
            out_layer_0, out_layer_1, out_layer_2 = data.get_activation_scores(lit_model)

            new_indices = sampling_class(out_layer_0, out_layer_1, out_layer_2, sample_size)

        elif sampling_method == "active_transfer_learning":
            """Implementation of active transfer learning for uncertainty sampling from 
            https://medium.com/pytorch/active-transfer-learning-with-pytorch-71ed889f08c1"""

            # get validation set predictions
            val_preds = torch.max(lit_model.val_predictions, dim=-1)[1]
            val_preds = val_preds.cpu().numpy()

            # new labels will be correct/incorrect
            new_labels = (val_preds == data.data_val.targets)
            new_labels = new_labels.astype(int)

            # get dataloaders
            new_train_dataloader, new_val_dataloader = _prepare_dataloaders(data.data_val.data, new_labels, data, args)

            # run fine-tuning and select samples that have highest probability of being incorrect
            new_indices = _finetune_and_sample(lit_model, data, new_train_dataloader, new_val_dataloader, sample_size, args)

        elif sampling_method == "DAL":
            """Implementation of discriminative active learning from https://arxiv.org/abs/1907.06347
            For speed up we only select a subsample of the larger dataset (train set or pool)"""

            # create equal sized sample from training set and unlabelled pool
            min_size = np.min([len(data.data_train.data), len(data.data_unlabelled.data)])
            if len(data.data_train.data) > len(data.data_unlabelled.data):
                subsample = np.random.choice(len(data.data_train.data), min_size, replace=False)
                new_train_data = data.data_train.data[subsample]
                new_pool_data = data.data_unlabelled.data
            elif len(data.data_train.data) < len(data.data_unlabelled.data):
                subsample = np.random.choice(len(data.data_unlabelled.data), min_size, replace=False)
                new_train_data = data.data_train.data
                new_pool_data = data.data_unlabelled.data[subsample]
            else:
                new_train_data = data.data_train.data
                new_pool_data = data.data_unlabelled.data
            assert len(new_train_data) == len(new_pool_data)

            # create dataset of training and unlabelled pool
            # 1 = labelled, 2 = unlabelled
            new_data = np.concatenate([new_train_data, new_pool_data])
            new_labels = [1]*len(new_train_data) + [0]*len(new_pool_data)
            new_labels = np.array(new_labels)

            # get dataloaders
            new_train_dataloader, new_val_dataloader = _prepare_dataloaders(new_data, new_labels, data, args)

            # run fine-tuning and select samples that have highest probability of being in the unlabelled pool
            new_indices = _finetune_and_sample(lit_model, data, new_train_dataloader, new_val_dataloader, sample_size, args)

        else:

            # reset predictions array of model
            lit_model.reset_predictions()

            # run a test loop so that we can get the model predictions and get model predictions
            trainer.test(lit_model, datamodule=data)
            predictions = lit_model.predictions  # maybe use a getPredictions method instead of referencing directly

            # get indices for samples to be labelled using the al_sampler methods
            new_indices = sampling_class(predictions, sample_size)

        if DEBUG_OUTPUT:
            print(f'Indices selected for labelling via method "{sampling_method}": \n-----------------\n')
            print(new_indices)
            print("\n-----------------\n")

        # adjust training set and unlabelled pool based on new queried indices
        data.expand_training_set(new_indices)
        unlabelled_data_size = data.get_ds_length(ds_name="unlabelled")

        al_iteration += 1
        
        if args.al_continue_training:
            # need to reset current epoch to 0 to not hit max_epochs directly in next training run
            trainer.current_epoch = 0
            print(f"Resetting current_epoch of trainer for {al_iteration}")

        else:
            # re-initialize model and trainer to start from scratch next iteration
            trainer, lit_model = _initialize_trainer(model_class, lit_model_class, data, args, logger, al_iteration)

    wandb.finish()


if __name__ == "__main__":
    main()
