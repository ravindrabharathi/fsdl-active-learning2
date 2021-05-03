"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb
import h5py
from PIL import Image


from active_learning import lit_models
from active_learning.data import al_sampler # for active learning sampling 
from torch.utils.data import ConcatDataset, DataLoader


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


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
    # Hide lines below until Lab 5
    parser.add_argument("--wandb", action="store_true", default=False)
    # Hide lines above until Lab 5
    parser.add_argument("--data_class", type=str, default="DroughtWatch")
    parser.add_argument("--model_class", type=str, default="ResnetClassifier")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--sampling_method", type=str, default='random')

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

    parser.add_argument("--help", "-h", action="help")
    return parser


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
    
    model = model_class(data_config=data.config(), args=args)
    sampling_method=args.sampling_method

    lit_model_class = lit_models.BaseLitModel
    

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    #logger = pl.loggers.TensorBoardLogger("training/logs")
    # Hide lines below until Lab 5
    #if args.wandb:
    project_name='fsdl-active-learning_'+sampling_method
    logger = pl.loggers.WandbLogger(name=project_name,project='fsdl-active-learning', job_type='train')
    logger.watch(model)
    logger.log_hyperparams(vars(args))
    # Hide lines above until Lab 5

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = None  # Don't Print full summary of the model # "full"
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")
    
    unlabelled_data_size=data.get_ds_length(ds_name='unlabelled')
    # pylint: disable=no-member
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    while (unlabelled_data_size>1000):

        

        trainer.fit(lit_model, datamodule=data)
        #reset predictions array of model 
        lit_model.reset_predictions()
        #run a test loop so that we can get the model predictions 
        trainer.test(lit_model, datamodule=data)
        #get model predictions 
        predictions = lit_model.predictions # maybe use a getPredictions method instead of referencing directly
        
        # now you can get indices for samples to be labelled using the al_sampler methods 

        # get random samples 
        # sample_size is the number of samples you need for labelling
        # pool_size is the size of the unlabelled pool size data.get_ds_length('unlabelled')
        
        if unlabelled_data_size>2000:
            sample_size=2000 # take 2000 extra samples at a time #int(0.25 * unlabelled_data_size)
        else:
            sample_size=unlabelled_data_size    
        print('Total Unlabelled Pool Size ', unlabelled_data_size)
        print('Query Sample size ',sample_size)

        if sampling_method=='random':
                
            
            new_indices= al_sampler.get_random_samples(pool_size=unlabelled_data_size,sample_size=sample_size)
            '''
            print('Random indices for labelling : \n-----------------\n') 
            print(new_indices)
            print('\n-----------------\n') 
            '''
        elif sampling_method=='least_confidence':
                
            # Get Least confidence samples 
            #pass predictions and sample size as args
            new_indices=al_sampler.get_least_confidence_samples(predictions,sample_size=sample_size)
            
            '''
            print('Least confidence query indices for labelling : \n-----------------\n') 
            print(new_indices) 
            print('\n-----------------\n')   
            '''

        elif sampling_method=='margin':
                

            # Get Top 2 Margin samples 
            #pass predictions and sample size as args
            new_indices=al_sampler.get_top2_confidence_margin_samples(predictions,sample_size=sample_size) 
            '''
            print('Top2 confidence margin samples : \n-------------------\n')
            print(new_indices)
            print('\n-----------------\n')
            '''
        elif sampling_method=='ratio':
                       
            new_indices=al_sampler.get_top2_confidence_ratio_samples(predictions,sample_size=sample_size) 

        elif sampling_method=='entropy':
                       
            new_indices=al_sampler.get_entropy_samples(predictions,sample_size=sample_size)     



        #adjust training set and unlabelled pool based on new queried indices 
        
        data.expand_training_set(new_indices)
        unlabelled_data_size=data.get_ds_length(ds_name='unlabelled')

        # pylint: enable=no-member
        '''
        # Hide lines below until Lab 5
        best_model_path = model_checkpoint_callback.best_model_path
        if best_model_path:
            print("Best model saved at:", best_model_path)
            if args.wandb:
                wandb.save(best_model_path)
                print("Best model also uploaded to W&B")
        # Hide lines above until Lab 5
        '''
    wandb.finish()    

if __name__ == "__main__":
    main()
