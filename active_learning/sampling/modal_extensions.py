"""
Implementation of various active learning techniques for the course FSDL Spring 2021 that can be used with modAL.
"""
from typing import Tuple
import torch
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
import hdbscan
from math import ceil

T_DEFAULT = 10
N_INSTANCES_DEFAULT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
DEBUG_OUTPUT = True
HDBSCAN_MIN_CLUSTER_SIZE = 500
OUTLIER_QUANTILE = 0.9
OUTLIER_FRACTION = 0.2
EMBEDDING_LAYER = "avgpool"
EMBEDDING_SIZE = 2048


def max_entropy(
    learner: ActiveLearner, X: np.array, n_instances: int = N_INSTANCES_DEFAULT, T: int = T_DEFAULT
) -> Tuple[torch.Tensor, np.array]:
    """Active learning sampling technique that maximizes the predictive entropy based on the paper
    'Deep Bayesian Active Learning with Image Data' (https://arxiv.org/pdf/1703.02910.pdf).

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = max_entropy, ...) # set max_entropy strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    T : int (default = 10)
      Number of predictions to generate per X instance from which then the entropy is estimated via taking the mean (see paper for details)

    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    return _batched_pytorch_tensor_loop(learner, X, n_instances, _max_entropy_scoring_function, T=T)


def bald(
    learner: ActiveLearner, X: np.array, n_instances: int = N_INSTANCES_DEFAULT, T: int = T_DEFAULT
) -> Tuple[torch.Tensor, np.array]:
    """Active learning sampling technique that maximizes the information gain via maximising mutual information between predictions
    and model posterior (Bayesian Active Learning by Disagreement - BALD) as depicted in the papers 'Deep Bayesian Active Learning
    with Image Data' (https://arxiv.org/pdf/1703.02910.pdf) and 'Bayesian Active Learning for Classification and Preference Learning'
    (https://arxiv.org/pdf/1112.5745.pdf).

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = bald, ...) # set bald strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    T : int (default = 10)
      Number of predictions to generate per X instance from which then the entropy is estimated via taking the mean (see paper for details)

    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    return _batched_pytorch_tensor_loop(learner, X, n_instances, _bald_scoring_function, T=T)


def random(
    learner: ActiveLearner, X: np.array, n_instances: int = N_INSTANCES_DEFAULT
) -> Tuple[torch.Tensor, np.array]:
    """Baseline active learning sampling technique that takes random instances from available pool.

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = random, ...) # set random strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]


def _batched_pytorch_tensor_loop(learner, X, n_instances, batch_scoring_function, **kwargs):

    if DEBUG_OUTPUT:
        print("Processing pool of instances with selected active learning strategy...")
        print("(Note: Based on the pool size this takes a while. Will generate debug output every 10%.)")
        ten_percent = int(len(X) / BATCH_SIZE / 10)
        i = 0
        percentage_output = 10

    # initialize pytorch tensor to store acquisition scores
    all_acquisitions = torch.Tensor().to(DEVICE)

    # create pytorch dataloader for batch-wise processing
    all_samples = torch.utils.data.DataLoader(X, batch_size=BATCH_SIZE)

    # process pool of instances batch wise
    for batch in all_samples:

        acquisitions = batch_scoring_function(learner, batch, **kwargs)

        all_acquisitions = torch.cat([all_acquisitions, acquisitions])

        if DEBUG_OUTPUT:
            i += 1
            if i > ten_percent:
                print(f"{percentage_output}% of samples in pool processed")
                percentage_output += 10
                i = 0

    # collect first n_instances to cpu and return
    idx = torch.argsort(-all_acquisitions)[:n_instances].cpu()
    return idx, X[idx]


def _bald_scoring_function(learner, batch, T):

    with torch.no_grad():

        outputs = torch.stack(
            [
                torch.softmax(  # probabilities from logits
                    learner.estimator.forward(batch, training=True, device=DEVICE), dim=-1
                )  # logits
                for t in range(T)  # multiple calculations to average over
            ]
        )

    mean_outputs = torch.mean(outputs, dim=0)

    H = torch.sum(-mean_outputs * torch.log(mean_outputs + 1e-10), dim=-1)
    E_H = -torch.mean(torch.sum(outputs * torch.log(outputs + 1e-10), dim=-1), dim=0)
    acquisitions = H - E_H

    return acquisitions


def _max_entropy_scoring_function(learner, batch, T):

    with torch.no_grad():

        outputs = torch.stack(
            [
                torch.softmax(  # probabilities from logits
                    learner.estimator.forward(batch, training=True, device=DEVICE), dim=-1
                )  # logits
                for t in range(T)  # multiple calculations to average over
            ]
        )

    mean_outputs = torch.mean(outputs, dim=0)

    acquisitions = torch.sum((-mean_outputs * torch.log(mean_outputs + 1e-10)), dim=-1)
    return acquisitions


def _featurize_instances(learner, X):
    if DEBUG_OUTPUT:
        print("Featurizing pool of instances for diversity sampling...")
        print("(Note: Based on the pool size this takes a while. Will generate debug output every 10%.)")
        ten_percent = int(len(X) / BATCH_SIZE / 10)
        i = 0
        percentage_output = 10

    # initialize pytorch tensor to store features
    all_features = torch.Tensor().to(DEVICE)

    # create pytorch dataloader for batch-wise processing
    all_samples = torch.utils.data.DataLoader(X, batch_size=BATCH_SIZE)

    # add hook to model that generates our features
    layer = learner.estimator.module.resnet._modules.get(EMBEDDING_LAYER)
    batch_features = torch.zeros(BATCH_SIZE, EMBEDDING_SIZE, 1, 1).to(DEVICE)

    def copy_data(m, i, o):
        batch_features.copy_(o.data)

    hook = layer.register_forward_hook(copy_data)

    # process pool of instances batch wise to featurize
    for batch in all_samples:

        # if last batch smaller: avoid size mismatch
        if batch.shape[0] < BATCH_SIZE:
            batch_features = torch.zeros(batch.shape[0], EMBEDDING_SIZE, 1, 1)

        # feed batch into network (get out features via registered hook)
        _ = learner.estimator.forward(batch, training=False, device=DEVICE)

        all_features = torch.cat([all_features, batch_features], dim=0)

        if DEBUG_OUTPUT:
            i += 1
            if i > ten_percent:
                print(f"{percentage_output}% of samples in pool featurized")
                percentage_output += 10
                i = 0

    hook.remove()

    all_features = all_features.squeeze().cpu().numpy()

    return all_features


def _cluster_instances(all_features, min_cluster_size):
    if DEBUG_OUTPUT:
        print("Performing clustering...")

    # perform clustering
    n_clusters = -1
    while n_clusters < 1:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(all_features)
        n_clusters = clusterer.labels_.max() + 1
        if n_clusters < 1:
            if DEBUG_OUTPUT:
                print(
                    f"HDBSCAN was not able to find clusters, halving min. cluster size from {min_cluster_size} to {int(min_cluster_size/2)}"
                )
            min_cluster_size = int(min_cluster_size / 2)

    return clusterer, n_clusters


def outlier(
    learner: ActiveLearner,
    X: np.array,
    n_instances: int = N_INSTANCES_DEFAULT,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
) -> Tuple[torch.Tensor, np.array]:
    """Active learning sampling technique that translates instances from the pool to features by using an internal layer of the
    learner model as featurizer, performing HDBSCAN to cluster the instances and then returning the outliers based on their
    GLOSH score.

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = outlier, ...) # set outlier strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    min_cluster_size : int (default = 100)
      Minimum number of points inside one cluster (used for HDBSCAN clustering algorithm)


    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    all_features = _featurize_instances(learner, X)

    clusterer, _ = _cluster_instances(all_features, min_cluster_size)

    # select indices with highest outlier scores and return them
    idx = np.argsort(-clusterer.outlier_scores_)[:n_instances]
    return idx, X[idx]


def cluster_outlier_combined(
    learner: ActiveLearner,
    X: np.array,
    n_instances: int = N_INSTANCES_DEFAULT,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
) -> Tuple[torch.Tensor, np.array]:
    """Active learning sampling technique that translates instances from the pool to features by using an internal layer of the
    learner model as featurizer, performing HDBSCAN to cluster the instances and then returning a combination of instances spread
    across all clusters and outliers detected via GLOSH.

    Examples
    --------
    >>> classifier = skorch.NeuralNetClassifier(MyModelClass, ...)
    >>> learner = modAL.models.ActiveLearner(estimator = classifier, query_strategy = cluster_outlier_combined, ...) # set cluster_outlier_combined strategy here
    >>> query_idx, query_instance = learner.query(sample_pool_X, ...) # strategy is then used here
    >>> learner.teach(X = sample_pool_X[query_idx], y = sample_pool_y[query_idx], ...)

    Parameters
    ----------
    learner : modAL.models.ActiveLearner
      modAL ActiveLearner instance with which the sampling technique should be used

    X : numpy.array
      Array of instances from which to sample from

    n_instances : int (default = 100)
      Number of instsances that should be sampled

    min_cluster_size : int (default = 100)
      Minimum number of points inside one cluster (used for HDBSCAN clustering algorithm)


    Returns
    -------
    Tuple of indexes and corresponding data instances that were chosen based on the sampling strategy.
    """

    all_features = _featurize_instances(learner, X)

    clusterer, n_clusters = _cluster_instances(all_features, min_cluster_size)

    # initialize variables needed below
    outlier_threshold = pd.Series(clusterer.outlier_scores_).quantile(OUTLIER_QUANTILE)
    n_features = len(all_features)
    instances_per_cluster = int(n_instances / n_clusters)
    all_selected_idx = np.array([]).astype(int)

    # check whether distribution between clusters can be done even - if not, distribute remaining equally
    if n_instances % n_clusters != 0:
        leftover_instances = n_instances - instances_per_cluster * n_clusters
    else:
        leftover_instances = 0

    if DEBUG_OUTPUT:
        print(f"{n_clusters} clusters found, entering loop to pick instances in all of them...")

    # loop over clusters to select instances in all of them
    for cluster_id in range(n_clusters):

        # distribute remaining instances if needed
        if leftover_instances > 0:
            remaining_clusters = n_clusters - cluster_id
            additional_instances = ceil((leftover_instances / remaining_clusters))
            instances_in_cluster = instances_per_cluster + additional_instances
            leftover_instances -= additional_instances
        else:
            instances_in_cluster = instances_per_cluster

        n_outliers = int(instances_in_cluster * OUTLIER_FRACTION)
        n_regular = instances_in_cluster - n_outliers

        # limit pool to current cluster only
        cluster_idx_all = np.where(clusterer.labels_ == cluster_id)[0]
        regular_idx_all = np.where(clusterer.outlier_scores_[cluster_idx_all] < outlier_threshold)[0]
        outlier_idx_all = np.where(clusterer.outlier_scores_[cluster_idx_all] >= outlier_threshold)[0]

        # avoid having a pool that is too small to sample from
        if len(regular_idx_all) < n_regular:
            leftover_instances += n_regular - len(regular_idx_all)
            n_regular = len(regular_idx_all)
        if len(outlier_idx_all) < n_outliers:
            leftover_instances += n_outliers - len(outlier_idx_all)
            n_outliers = len(outlier_idx_all)

        # select both regular and outlier instances for this cluster
        regular_idx_selected = np.random.choice(len(regular_idx_all), size=n_regular, replace=False)
        outlier_idx_selected = np.random.choice(len(outlier_idx_all), size=n_outliers, replace=False)

        # calculate all idx that are selected from this cluster
        if n_outliers > 0:
            outlier_idx_original = np.array(range(n_features))[cluster_idx_all][outlier_idx_all][
                outlier_idx_selected
            ]  # outlier idx of this cluster (idx translated to full pool)
        else:
            outlier_idx_original = np.array([]).astype(int)

        if n_regular > 0:
            regular_idx_original = np.array(range(n_features))[cluster_idx_all][regular_idx_all][
                regular_idx_selected
            ]  # regular idx of this cluster (idx translated to full pool)
        else:
            regular_idx_original = np.array([]).astype(int)

        cluster_idx_selected = np.concatenate([outlier_idx_original, regular_idx_original])

        all_selected_idx = np.concatenate([all_selected_idx, cluster_idx_selected])

        if DEBUG_OUTPUT:
            print(f"Instance selection on cluster no. {cluster_id} done.")

    if len(all_selected_idx) < n_instances:
        print(
            f"CAUTION: cluster_outlier_combined algorithm was not able to build a set of {n_instances} instances as requested, instead only returning {len(all_selected_idx)} instances."
        )

    return all_selected_idx, X[all_selected_idx]
