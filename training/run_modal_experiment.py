"""modAL experiment-running framework."""
import argparse
import importlib
import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from modAL.models import ActiveLearner
import skorch
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from active_learning import lit_models

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
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
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
    parser.add_argument("--al_epochs_init", type=int, default=20, help="No. of initial epochs to train before beginning with active learning")
    parser.add_argument("--al_epochs_incr", type=int, default=20, help="No. of epochs to train in each active learning iteration")
    parser.add_argument("--al_n_iter", type=int, default=33, help="No. of active learning iterations")
    parser.add_argument("--al_samples_per_iter", type=int, default=2000, help="No. of samples to query per active learning iteration")
    parser.add_argument("--al_incr_onlynew", type=bool, default=False, help="Whether to take only newly queried samples in each active learning iterations, or the full training data")
    parser.add_argument("--al_query_strategy", type=str, choices=["uncertainty_sampling", "margin_sampling", "entropy_sampling", "max_entropy", "bald", "random", "outlier", "cluster_outlier_combined"], default="uncertainty_sampling", help="Active learning query strategy")

    # For debugging / development purposes
    parser.add_argument("--reduced_develop_train_size", type=bool, default=False, help="Whether to take only a very small set to train (allows for faster results during development)")

    parser.add_argument("--help", "-h", action="help")
    return parser

def _log_skorch_history(history: skorch.history.History, al_iter: int, epoch_start: int, train_acc: float, train_size: int, wandb_logging: bool = True) -> None:

    for epoch, train_loss, valid_loss, valid_acc, dur in history[:, ('epoch', 'train_loss', 'valid_loss', 'valid_acc', 'dur')]:

        metrics = {
            'epoch': epoch_start + epoch, 
            'train_loss': train_loss, 
            'valid_loss': valid_loss, 
            'valid_acc': valid_acc, 
            'dur': dur,
            'al_iter': al_iter, 
        }
                
        # add train_acc to last item of history
        if epoch == len(history[:, 'train_loss']):
            metrics['train_acc'] = train_acc

        #print(metrics)
        if wandb_logging:
            wandb.log(metrics)


def main():
    """
    Run an active learning experiment.

    Sample command:
    ```
    python training/run_modAL_experiment.py --al_epochs_init=10 --al_epochs_incr=5 --al_n_iter=10 --al_samples_per_iter=100 --data_class=DroughtWatch --model_class=ResnetClassifier --batch_size=64 --n_train_images=1000 --n_validation_images=1000 --pretrained=True --wandb 
    ```
    """

    # generic setup steps from run_experiment
    # ---------------------------------------

    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"active_learning.data.{args.data_class}")
    model_class = _import_class(f"active_learning.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseLitModel

    if args.loss == "ctc":
        lit_model_class = lit_models.CTCLitModel

    if args.loss == "transformer":
        lit_model_class = lit_models.TransformerLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")


    # modAL specific experiment setup
    # -------------------------------

    # initialize wandb with pytorch model
    if args.wandb:
        wandb.init(config=args)
        wandb.watch(model, log_freq=100)

    # evaluate query strategy from args parameter
    if args.al_query_strategy in ["uncertainty_sampling", "margin_sampling", "entropy_sampling"]:
        query_strategy = _import_class(f"modAL.uncertainty.{args.al_query_strategy}")
    else:
        query_strategy = _import_class(f"active_learning.sampling.{args.al_query_strategy}")

    # cpu vs. gpu: ignore --gpu args param, instead just set gpu based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu" 

     # initialize train, validation and pool datasets
    data.setup()

    X_initial = np.moveaxis(data.data_train.data, 3, 1) # shape change: (i, channels, h, w) instead of (i, h, w, channels)
    y_initial = data.data_train.targets
    if args.reduced_develop_train_size:
        print("NOTE: Reduced initial train set size for development activated")
        X_initial = X_initial[:100, :, :, :]
        y_initial = y_initial[:100]

    X_val = np.moveaxis(data.data_val.data, 3, 1) # shape change
    y_val = data.data_val.targets
    X_pool = np.moveaxis(data.data_unlabelled.data, 3, 1) # shape change
    y_pool = data.data_unlabelled.targets

    # initialize skorch classifier
    classifier = NeuralNetClassifier(model,
                                     criterion=torch.nn.CrossEntropyLoss,
                                     optimizer=torch.optim.Adam,
                                     train_split=predefined_split(Dataset(X_val, y_val)),
                                     verbose=1,
                                     device=device,
                                     )

    lit_model.summarize(mode="full")

    # initialize modal active learner
    print("Initializing model with base training set")
    learner = ActiveLearner(
        estimator=classifier,
        X_training=X_initial, 
        y_training=y_initial, 
        epochs=args.al_epochs_init,
        query_strategy=query_strategy
    )

    _log_skorch_history(
        history = learner.estimator.history, 
        al_iter = 0, 
        epoch_start = 0, 
        train_acc = learner.score(learner.X_training, learner.y_training),
        train_size = len(learner.y_training),
        wandb_logging = args.wandb)

    # active learning loop
    for idx in range(args.al_n_iter):

        print('Active learning query no. %d' % (idx + 1))
        query_idx, _ = learner.query(X_pool, n_instances=args.al_samples_per_iter)
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx], only_new=args.al_incr_onlynew, epochs=args.al_epochs_incr
        )

        _log_skorch_history(
            history = learner.estimator.history, 
            al_iter = idx+1, 
            epoch_start = args.al_epochs_init + idx*args.al_epochs_incr, 
            train_acc = learner.score(learner.X_training, learner.y_training),
            train_size = len(learner.y_training),
            wandb_logging = args.wandb)

        # remove queried instances from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)


if __name__ == "__main__":
    main()
