import numpy as np
import torch


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
    margins = max_probs[:,0] - max_probs[:,1]
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
    scores = max_probs[:,1] / max_probs[:,0]

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
    margins = max_probs[:,0] - max_probs[:,1]
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
    scores = max_probs[:,1] / max_probs[:,0]

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


def mb_outliers_mean(out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int) -> np.array:
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


def mb_outliers_max(out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int) -> np.array:
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


def mb_outliers_mean_least_confidence(out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int) -> np.array:
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
    pre_sample_size = 4*sample_size
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


def mb_outliers_mean_entropy(out_layer_1: torch.Tensor, out_layer_2: torch.Tensor, out_layer_3: torch.Tensor, sample_size: int) -> np.array:
    """Combined uncertainty and diversity sampling technique from https://towardsdatascience.com/advanced-active-learning-cheatsheet-d6710cba7667"""

    # Part 1: pre-sample with least confidence

    # convert final layer to probabilities 
    probabilities = torch.nn.functional.softmax(out_layer_3, dim=-1)
    # calculate entropy
    probslogs = probabilities * torch.log2(probabilities)
    summed = probslogs.sum(dim=1)
    entropy_scores = torch.subtract(torch.zeros_like(summed), summed) / np.log2(probslogs.size()[1])

    # define pre_sample_size
    pre_sample_size = 4*sample_size
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