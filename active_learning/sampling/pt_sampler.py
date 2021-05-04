import torch
import numpy as np


def get_random_samples(pool_size, sample_size):
    
    # if pool size smaller than sample size, return entire pool
    sample_size = np.min([pool_size, sample_size])
    # get random indixes from unlabelled pool
    idxs = np.random.choice(np.arange(pool_size), sample_size, replace=False)

    return idxs


def get_least_confidence_samples(predictions, sample_size):
        
    # get top probabilities
    max_probs = torch.max(predictions, dim=-1)[0]
    # get number of classes
    num_classes = probs[0].shape[0]
    # calculate least confidence scores
    scores = (num_classes * (1 - max_probs)) / (num_classes - 1)

    # pick k examples with highest uncertainty scores
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def get_top2_confidence_margin_samples(predictions, sample_size):
        
    # get two highest probabilities
    max_probs, _ = torch.topk(predictions, 2, largest=True, dim=1)
    # calculate margins between two most confident predictions
    margins = max_probs[:,0] - max_probs[:,1]
    # calculate scores between 0 and 1
    scores = 1 - margins

    # pick k examples with smallest margins
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def get_top2_confidence_ratio_samples(predictions, sample_size):

    # get two highest probabilities
    max_probs, _ = torch.topk(predictions, 2, largest=True, dim=1)
    # calculate ratio between two most confident predictions
    scores = max_probs[:,1] / max_probs[:,0]

    # pick k examples with highest ratios
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs


def get_entropy_samples(predictions, sample_size):

    # calculate entropy
    probslogs = predictions * torch.log2(predictions)
    summed = probslogs.sum(dim=1)
    numerator = torch.subtract(torch.zeros_like(summed), summed)
    denominator = np.log2(probslogs.size()[1])
    scores = numerator/denominator

    # pick k examples with highest entropy scores
    _, idxs = torch.topk(scores, sample_size, largest=True)
    idxs = idxs.detach().cpu().numpy()

    return idxs

