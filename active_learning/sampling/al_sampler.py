from math import ceil, isnan

import hdbscan
import numpy as np
import torch

DEBUG_OUTPUT = True


def random(predictions, sample_size):

    pool_size = len(predictions)

    # return sample size of random indices
    if pool_size <= 2000:
        indices = [x for x in range(pool_size)]  # return complete pool size
    else:
        indices = np.random.choice(np.arange(pool_size), sample_size, replace=False)

    return indices


def least_confidence(predictions, sample_size):

    conf = []
    indices = []

    for idx, prediction in enumerate(predictions):
        most_confident = np.max(prediction)
        n_classes = prediction.shape[0]
        conf_score = (1 - most_confident) * n_classes / (n_classes - 1)
        conf.append(conf_score)
        indices.append(idx)

    conf = np.asarray(conf)
    indices = np.asarray(indices)
    result = indices[np.argsort(conf)][:sample_size]

    return result


def margin(predictions, sample_size):

    margins = []
    indices = []

    for idx, predxn in enumerate(predictions):
        predxn[::-1].sort()
        margin = predxn[0] - predxn[1]
        margins.append(margin)
        indices.append(idx)
    margins = np.asarray(margins)
    indices = np.asarray(indices)
    least_margin_indices = indices[np.argsort(margins)][:sample_size]

    return least_margin_indices


def ratio(predictions, sample_size):

    margins = []
    indices = []

    for idx, predxn in enumerate(predictions):
        predxn[::-1].sort()
        margins.append(predxn[1] / predxn[0])
        indices.append(idx)
    margins = np.asarray(margins)
    indices = np.asarray(indices)
    confidence_ratio_indices = indices[np.argsort(margins)][:sample_size]

    return confidence_ratio_indices


def entropy(predictions, sample_size):

    entropies = []
    indices = []

    for idx, predxn in enumerate(predictions):
        log2p = np.log2(predxn)
        pxlog2p = predxn * log2p
        n = len(predxn)
        entropy = -np.sum(pxlog2p) / np.log2(n)
        entropies.append(entropy)
        indices.append(idx)
    entropies = np.asarray(entropies)
    indices = np.asarray(indices)
    max_entropy_indices = np.argsort(entropies)[-sample_size:]

    return max_entropy_indices


def bald(probabilities: torch.Tensor, sample_size: int) -> np.array:
    """Active learning sampling technique that maximizes the information gain via maximising mutual information between predictions
    and model posterior (Bayesian Active Learning by Disagreement - BALD) as depicted in the papers 'Deep Bayesian Active Learning
    with Image Data' (https://arxiv.org/pdf/1703.02910.pdf) and 'Bayesian Active Learning for Classification and Preference Learning'
    (https://arxiv.org/pdf/1112.5745.pdf).
    """

    mean_probabilities = torch.mean(probabilities, dim=-1)

    H = torch.sum(-mean_probabilities * torch.log(mean_probabilities + 1e-10), dim=-1)
    E_H = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1), dim=-1)

    scores = H - E_H

    idx = torch.argsort(-scores)[:sample_size].cpu().numpy()
    return idx


def max_entropy(probabilities: torch.Tensor, sample_size: int) -> np.array:
    """Active learning sampling technique that maximizes the predictive entropy based on the paper 'Deep Bayesian Active Learning
    with Image Data' (https://arxiv.org/pdf/1703.02910.pdf).
    """

    mean_probabilities = torch.mean(probabilities, dim=2)

    scores = torch.sum((-mean_probabilities * torch.log(mean_probabilities + 1e-10)), dim=-1)

    idx = torch.argsort(-scores)[:sample_size].cpu().numpy()
    return idx


def least_confidence_pt(predictions, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b"""

    # convert np array to torch tensor
    probabilities = torch.Tensor(predictions)

    # get top probabilities
    max_probs = torch.max(probabilities, dim=-1)[0]
    # get number of classes
    num_classes = probabilities[0].shape[0]
    # calculate least confidence scores
    scores = (num_classes * (1 - max_probs)) / (num_classes - 1)

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with highest uncertainty scores
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def margin_pt(predictions, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b"""

    # convert np array to torch tensor
    probabilities = torch.Tensor(predictions)

    # get two highest probabilities
    max_probs, _ = torch.topk(probabilities, 2, largest=True, dim=1)
    # calculate margins between two most confident predictions
    margins = max_probs[:, 0] - max_probs[:, 1]
    # calculate scores between 0 and 1
    scores = 1 - margins

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with smallest margins
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def ratio_pt(predictions, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b"""

    # convert np array to torch tensor
    probabilities = torch.Tensor(predictions)

    # get two highest probabilities
    max_probs, _ = torch.topk(probabilities, 2, largest=True, dim=1)
    # calculate ratio between two most confident predictions
    scores = max_probs[:, 1] / max_probs[:, 0]

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with highest ratios
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def entropy_pt(predictions, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b"""

    # convert np array to torch tensor
    probabilities = torch.Tensor(predictions)

    # calculate entropy
    probslogs = probabilities * torch.log2(probabilities)
    summed = probslogs.sum(dim=1)
    scores = torch.subtract(torch.zeros_like(summed), summed) / np.log2(probslogs.size()[1])

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with highest entropy scores
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def least_confidence_mc(probabilities: torch.Tensor, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
    adjusted for Bayesian Active Learning"""

    # calculate average probabilities from multiple inference runs
    probabilities = torch.mean(probabilities, dim=2)

    # get top probabilities
    max_probs = torch.max(probabilities, dim=-1)[0]
    # get number of classes
    num_classes = probabilities[0].shape[0]
    # calculate least confidence scores
    scores = (num_classes * (1 - max_probs)) / (num_classes - 1)

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with highest uncertainty scores
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def margin_mc(probabilities: torch.Tensor, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
    adjusted for Bayesian Active Learning"""

    # calculate average probabilities from multiple inference runs
    probabilities = torch.mean(probabilities, dim=2)

    # get two highest probabilities
    max_probs, _ = torch.topk(probabilities, 2, largest=True, dim=1)
    # calculate margins between two most confident predictions
    margins = max_probs[:, 0] - max_probs[:, 1]
    # calculate scores between 0 and 1
    scores = 1 - margins

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with smallest margins
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def ratio_mc(probabilities: torch.Tensor, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
    adjusted for Bayesian Active Learning"""

    # calculate average probabilities from multiple inference runs
    probabilities = torch.mean(probabilities, dim=2)

    # get two highest probabilities
    max_probs, _ = torch.topk(probabilities, 2, largest=True, dim=1)
    # calculate ratio between two most confident predictions
    scores = max_probs[:, 1] / max_probs[:, 0]

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with highest ratios
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def entropy_mc(probabilities: torch.Tensor, sample_size: int) -> np.array:
    """Uncertainty sampling technique from https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b
    adjusted for Bayesian Active Learning"""

    # calculate average probabilities from multiple inference runs
    probabilities = torch.mean(probabilities, dim=2)

    # calculate entropy
    probslogs = probabilities * torch.log2(probabilities)
    summed = probslogs.sum(dim=1)
    scores = torch.subtract(torch.zeros_like(summed), summed) / np.log2(probslogs.size()[1])

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with highest entropy scores
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def mb_outliers_mean(
    out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int
) -> np.array:
    """Diversity sampling technique from https://towardsdatascience.com/https-towardsdatascience-com-diversity-sampling-cheatsheet-32619693c304"""

    # calculate mean activations for each layer's activations
    mean_layer_1 = torch.mean(out_layer_1, dim=-1)
    mean_layer_2 = torch.mean(out_layer_2, dim=-1)
    mean_layer_3 = torch.mean(out_layer_3, dim=-1)

    # get average scores across layers
    scores = (mean_layer_1 + mean_layer_2 + mean_layer_3) / 3

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with lowest activation scores
    _, idxs = torch.topk(scores, sample_size, largest=False)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def mb_outliers_max(
    out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int
) -> np.array:
    """Diversity sampling technique from https://towardsdatascience.com/https-towardsdatascience-com-diversity-sampling-cheatsheet-32619693c304"""

    # calculate max activations for each layer's activations
    max_layer_1 = torch.max(out_layer_1, dim=-1).values
    max_layer_2 = torch.max(out_layer_2, dim=-1).values
    max_layer_3 = torch.max(out_layer_3, dim=-1).values

    # get overall maximum activation value
    scores = torch.max(torch.max(max_layer_1, max_layer_2), max_layer_3)

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with lowest activation scores
    _, idxs = torch.topk(scores, sample_size, largest=False)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def mb_outliers_mean_least_confidence(
    out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int
) -> np.array:
    """Combined uncertainty and diversity sampling technique from https://towardsdatascience.com/advanced-active-learning-cheatsheet-d6710cba7667"""

    # Part 1: pre-sample with least confidence

    # convert final layer to probabilities
    probabilities = torch.nn.functional.softmax(out_layer_3, dim=-1)
    # get top probabilities
    max_probs = torch.max(probabilities, dim=-1)[0]
    # get number of classes
    num_classes = probabilities[0].shape[0]
    # calculate least confidence scores
    least_conf_scores = (num_classes * (1 - max_probs)) / (num_classes - 1)

    # define pre_sample_size
    pre_sample_size = 4 * sample_size
    # make sure sample size doesn't exceed scores
    pre_sample_size = np.min([len(least_conf_scores), pre_sample_size])

    # pick k examples with highest uncertainty scores
    _, least_conf_idxs = torch.topk(least_conf_scores, pre_sample_size, largest=True)
    least_conf_idxs = least_conf_idxs.detach().cpu().numpy()

    # Part 2: sample from lease confidence pool using model based outliers

    # calculate mean activations for each layer's activations
    mean_layer_1 = torch.mean(out_layer_1, dim=-1)
    mean_layer_2 = torch.mean(out_layer_2, dim=-1)
    mean_layer_3 = torch.mean(out_layer_3, dim=-1)

    # get average scores across layers
    scores_raw = (mean_layer_1 + mean_layer_2 + mean_layer_3) / 3

    # select most uncertain examples
    scores = scores_raw[least_conf_idxs]

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with lowest activation scores
    _, idxs = torch.topk(scores, sample_size, largest=False)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def mb_outliers_mean_entropy(
    out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int
) -> np.array:
    """Combined uncertainty and diversity sampling technique from https://towardsdatascience.com/advanced-active-learning-cheatsheet-d6710cba7667"""

    # Part 1: pre-sample with least confidence

    # convert final layer to probabilities
    probabilities = torch.nn.functional.softmax(out_layer_3, dim=-1)
    # calculate entropy
    probslogs = probabilities * torch.log2(probabilities)
    summed = probslogs.sum(dim=1)
    entropy_scores = torch.subtract(torch.zeros_like(summed), summed) / np.log2(probslogs.size()[1])

    # define pre_sample_size
    pre_sample_size = 4 * sample_size
    # make sure sample size doesn't exceed scores
    pre_sample_size = np.min([len(entropy_scores), pre_sample_size])

    # pick k examples with highest uncertainty scores
    _, entropy_idxs = torch.topk(entropy_scores, pre_sample_size, largest=True)
    entropy_idxs = entropy_idxs.detach().cpu().numpy()

    # Part 2: sample from lease confidence pool using model based outliers

    # calculate mean activations for each layer's activations
    mean_layer_1 = torch.mean(out_layer_1, dim=-1)
    mean_layer_2 = torch.mean(out_layer_2, dim=-1)
    mean_layer_3 = torch.mean(out_layer_3, dim=-1)

    # get average scores across layers
    scores = (mean_layer_1 + mean_layer_2 + mean_layer_3) / 3

    # select most uncertain examples
    scores = scores[entropy_idxs]

    # make sure sample size doesn't exceed scores
    sample_size = np.min([len(scores), sample_size])

    # pick k examples with lowest activation scores
    _, idxs = torch.topk(scores, sample_size, largest=False)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def mb_clustering(
    out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int
) -> np.array:
    """Active learning sampling technique that translates instances from the pool to features by using an internal layer of the
    learner model as featurizer, performing HDBSCAN to cluster the instances and then returning a combination of instances spread
    across all clusters and outliers detected via GLOSH.
    """

    HDBSCAN_CLUSTER_OUTLIER_QUANTILE = 0.9
    HDBSCAN_CLUSTER_OUTLIER_FRACTION = 0.2

    # model based featurizing
    features = out_layer_3.cpu().numpy()

    # cluster and calculate glosh outlier scores
    clusterer, n_clusters = _hdbscan_cluster_instances(features)

    # initialize variables needed below
    n_features = len(features)
    instances_per_cluster = int(sample_size / n_clusters)
    all_selected_idx = np.array([]).astype(int)

    # check whether distribution between clusters can be done even - if not, distribute remaining equally
    if sample_size % n_clusters != 0:
        leftover_instances = sample_size - instances_per_cluster * n_clusters
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

        n_outliers = int(instances_in_cluster * HDBSCAN_CLUSTER_OUTLIER_FRACTION)
        n_regular = instances_in_cluster - n_outliers

        # limit pool to current cluster only
        # NOTE: possible RuntimeWarning because of NaN values (https://github.com/scikit-learn-contrib/hdbscan/issues/374)
        cluster_idx_all = np.where(clusterer.labels_ == cluster_id)[0]

        # divide into main points and outliers
        outlier_threshold = np.quantile(clusterer.outlier_scores_[cluster_idx_all], HDBSCAN_CLUSTER_OUTLIER_QUANTILE)
        if not (isnan(outlier_threshold) or outlier_threshold == 0):
            regular_idx_all = np.where(clusterer.outlier_scores_[cluster_idx_all] < outlier_threshold)[0]
            outlier_idx_all = np.where(clusterer.outlier_scores_[cluster_idx_all] >= outlier_threshold)[0]

        #  if possible cluster cannot be devided into main points and outliers, just sample randomly
        else:
            n_cluster_points = len(cluster_idx_all)
            outlier_idx_all = np.random.choice(
                range(n_cluster_points), int(n_cluster_points * HDBSCAN_CLUSTER_OUTLIER_FRACTION), replace=False
            )
            regular_idx_all = np.delete(range(n_cluster_points), outlier_idx_all)

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

    if len(all_selected_idx) < sample_size:
        print(
            f"CAUTION: cluster_outlier_combined algorithm was not able to build a set of {sample_size} instances as requested because HDBSCAN clusters did not contain enough points."
        )
        print(f"Selecting {sample_size-len(all_selected_idx)} random instances additionally.")

        remaining_idx = np.delete(range(len(clusterer.labels_)), all_selected_idx)
        all_selected_idx = np.concatenate(
            [all_selected_idx, np.random.choice(remaining_idx, sample_size - len(all_selected_idx), replace=False)]
        )

    return all_selected_idx


def mb_outliers_glosh(
    out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int
) -> np.array:
    """Active learning sampling technique that translates instances from the pool to features by using an internal layer of the
    model as featurizer, performing HDBSCAN to cluster the instances and then returning the outliers based on their GLOSH score.
    """

    # model based featurizing
    features = out_layer_3.cpu().numpy()

    # cluster and calculate glosh outlier scores
    clusterer, n_clusters = _hdbscan_cluster_instances(features)

    if DEBUG_OUTPUT:
        print(f"{n_clusters} clusters found, returning outliers based on max. GLOSH score")

    # select indices with highest outlier scores and return them
    # NOTE: prints a RuntimeWarning sometimes because of NaN values (https://github.com/scikit-learn-contrib/hdbscan/issues/374)
    idx = np.argsort(-clusterer.outlier_scores_)[:sample_size]
    return idx


def _hdbscan_cluster_instances(features):

    HDBSCAN_MIN_CLUSTER_SIZE = 100

    if DEBUG_OUTPUT:
        print(f"Performing HDBSCAN clustering for pool with shape {features.shape}...")

    # perform clustering
    n_clusters = -1

    min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE

    while n_clusters < 1:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=1).fit(features)
        n_clusters = clusterer.labels_.max() + 1
        if n_clusters < 1:
            if DEBUG_OUTPUT:
                print(
                    f"HDBSCAN was not able to find any clusters, halving min. cluster size from {min_cluster_size} to {int(min_cluster_size/2)}"
                )
            min_cluster_size = int(min_cluster_size / 2)

    return clusterer, n_clusters
