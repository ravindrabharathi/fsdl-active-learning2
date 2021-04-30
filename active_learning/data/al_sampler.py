import numpy as np
import torch

def get_random_samples(pool_size,sample_size):
    
    #return sample size of random indices 
    if pool_size<=2000:
        indices=[x for x in range(pool_size)]  #return complete pool size 
    else:
        indices=np.random.randint(pool_size, size=sample_size)

    return indices

def get_least_confidence_samples(predictions, sample_size):

    conf = []
    indices = []
    for idx,prediction in enumerate(predictions):
        most_confident = np.max(prediction)
        n_classes=prediction.shape[0]
        conf_score=(1-most_confident)*n_classes/(n_classes-1)
        conf.append(conf_score)
        indices.append(idx)
            
    conf = np.asarray(conf)
    indices = np.asarray(indices)
    result=indices[np.argsort(conf)][:sample_size]
    
        
    return result

def get_top2_confidence_margin_samples(predictions, sample_size):

    
    margins = []
    indices = []
        
    for idx,predxn in enumerate(predictions):
        predxn[::-1].sort()
        margin=predxn[0]-predxn[1]
        margins.append(margin)
        indices.append(idx)
    margins=np.asarray(margins)
    indices=np.asarray(indices)
    least_margin_indices=indices[np.argsort(margins)][:sample_size]
  
    return least_margin_indices

def get_top2_confidence_ratio_samples(predictions, sample_size):

    
    margins = []
    indices = []
        
    for idx,predxn in enumerate(predictions):
        predxn[::-1].sort()
        margins.append(predxn[1]/predxn[0])
        indices.append(idx)
    margins=np.asarray(margins)
    indices=np.asarray(indices)
    confidence_ratio_indices=indices[np.argsort(margins)][:sample_size]
  
    return confidence_ratio_indices 

def get_entropy_samples(predictions,sample_size):
    entropies = []
    indices = []
    for idx,predxn in enumerate(predictions):
        log2p=np.log2(predxn)
        pxlog2p=predxn * log2p
        n=len(predxn)
        entropy=-np.sum(pxlog2p)/np.log2(n)
        entropies.append(entropy)
        indices.append(idx)
    entropies=np.asarray(entropies)
    indices=np.asarray(indices)
    max_entropy_indices=np.argsort(entropies)[-sample_size:]  
    return max_entropy_indices     