import torch
import torch.nn as nn

def infoNCELoss(scores, labels, temperature=0.1):
    """
    Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
    We extraly weigh the positive samples using the ground truth likelihood on those positions
    
    loss = - 1/sum(weights) * sum(inner_element*weights)
    inner_element = log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
    """
    
    exp_scores = torch.exp(scores / temperature)
    bool_mask = labels>0 # only keep positive samples, we set a threshod on the likelihood in GT    
    
    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    inner_element = torch.log(torch.masked_select(exp_scores/denominator, bool_mask))
    loss = -torch.sum(inner_element*torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))
    
    return loss


def cross_entropy_loss(logits, labels):
    return -torch.sum(labels * nn.LogSoftmax(dim=1)(logits)) / logits.size()[0]