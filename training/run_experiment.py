"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from active_learning import lit_models
from active_learning.lit_models.base import MaxAccuracyLogger

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)

DEBUG_OUTPUT = True
MC_ITERATIONS = 10

BASIC_SAMPLING_METHODS = ["random", "least_confidence", "margin", "ratio", "entropy", "least_confidence_pt", "margin_pt", "ratio_pt", "entropy_pt"]
MC_SAMPLING_METHODS = ["bald", "max_entropy", "least_confidence_mc", "margin_mc", "ratio_mc", "entropy_mc"]
MB_SAMPLING_METHODS = ["mb_outliers_mean", "mb_outliers_max", "mb_outliers_mean_least_confidence", "mb_outliers_mean_entropy", "mb_outliers_glosh", "mb_clustering"]


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
    parser.add_argument("--sampling_method", type=str, choices=BASIC_SAMPLING_METHODS+MC_SAMPLING_METHODS+MB_SAMPLING_METHODS, default="random", help="Active learning sampling strategy")
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
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=5)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    max_accuracy_callback = MaxAccuracyLogger()
    callbacks = [early_stopping_callback, model_checkpoint_callback, max_accuracy_callback]

    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    return trainer, lit_model


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
    sampling_class = _import_class(f"active_learning.sampling.al_sampler.{sampling_method}")

    lit_model_class = lit_models.BaseLitModel

    # --wandb args parameter ignored for now, always using wandb
    # specify project & entity outside of code, example: wandb init --project fdsl-active-learning --entity fsdl_active_learners
    project_name = "fsdl-active-learning_" + sampling_method
    logger = pl.loggers.WandbLogger(name=project_name, job_type="train") 
    logger.log_hyperparams(vars(args))

    args.weights_summary = None  # set to "full" to print model layer summary
    unlabelled_data_size = data.get_ds_length(ds_name="unlabelled")

    al_iteration = 0

    trainer, lit_model = _initialize_trainer(model_class, lit_model_class, data, args, logger, al_iteration)
    
    while unlabelled_data_size > 0 and (args.al_iter < 0 or al_iteration < args.al_iter):

        # fit model on current data
        trainer.fit(lit_model, datamodule=data)

        print(f"callback_metrics after fit of iteration {al_iteration}: {trainer.callback_metrics}")

        # log best accuracies of this iteration to wandb
        wandb.log({
            "train_size": data.get_ds_length(ds_name="train"),
            "train_acc_best": trainer.callback_metrics["train_acc_max"],
            "val_acc_best": trainer.callback_metrics["val_acc_max"],
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
