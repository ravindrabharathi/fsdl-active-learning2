import numpy as np
import torch


def random(predictions, sample_size):

    pool_size = len(predictions)

    # return sample size of random indices
    if pool_size <= 2000:
        indices = [x for x in range(pool_size)]  # return complete pool size
    else:
        indices = np.random.randint(pool_size, size=sample_size)

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
