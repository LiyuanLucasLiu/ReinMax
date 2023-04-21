# The code has been modified from https://github.com/chijames/GST

import torch
import torch.nn.functional as F

from reinmax import reinmax as reinmax_kernel

def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1)
    return one_hot

def apply_mask(logits, mask):
    ddif = len(logits.shape) - len(mask.shape)
    mask = mask.view(mask.shape + (1,)*ddif) if ddif>0 else mask
    logits.masked_fill_(~mask, float('-inf'))
    mask_sum = (mask.sum(dim=-1, keepdim=True) == 0)
    logits.masked_fill_(mask_sum, 1.)
    return logits 
        
def masked_softmax(logits, mask=None, dim=-1):
    if mask is not None:
        logits = apply_mask(logits, mask)
    probs = F.softmax(logits, dim=dim)
    return probs

def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot

def st(logits, tau, mask=None):
    
    if mask is not None:
        logits = apply_mask(logits, mask)
        
    shape, dtype = logits.size(), logits.dtype
    logits = logits.view(-1, shape[-1]).float()
    
    y_soft = logits.softmax(dim=-1)
    sample = torch.multinomial(
        y_soft,
        num_samples=1,
        replacement=True,
    )
    one_hot_sample = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)
    
    prob = (logits/tau).softmax(dim=-1)
    one_hot_sample = one_hot_sample - prob.detach() + prob
    return one_hot_sample.view(shape), y_soft.view(shape)
        
def reinmax(logits, tau=1.0, mask=None):
    if mask is not None:
        logits = apply_mask(logits, mask)
    grad_sample, y_soft = reinmax_kernel(logits, tau)
    return grad_sample, y_soft

def gst_mover(logits, tau=1.0, mask=None, hard=True, gap=1.0, detach=True):
    logits_cpy = logits.detach() if detach else logits
    probs = masked_softmax(logits_cpy)
    if mask is not None:
        mask = mask.float()
        ddif = len(probs.shape) - len(mask.shape)
        mask = mask.view(mask.shape + (1,)*ddif) if ddif>0 else mask
        probs = probs * mask + 1e-20
        probs = probs / probs.sum(-1, keepdim=True)
        
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs)  
    action = m.sample() 
    argmax = probs.argmax(dim=-1, keepdim=True)
    
    action_bool = action.bool()
    max_logits = torch.gather(logits_cpy, -1, argmax)
    move = (max_logits - logits_cpy)*action
    
    if type(gap)!=float:
        pi = probs[action_bool]*(1-1e-5) # for numerical stability
        gap = ( -(1-pi).log()/pi ).view( logits.shape[:-1] + (1,) )
    move2 = ( logits_cpy + (-max_logits + gap) ).clamp(min=0.0)
    move2[action_bool] = 0.0
    logits = logits + (move - move2)

    logits = logits - logits.mean(dim=-1, keepdim=True)
    probs = masked_softmax(logits=logits / tau)
    if mask is not None:
        mask = mask.float()
        ddif = len(probs.shape) - len(mask.shape)
        mask = mask.view(mask.shape + (1,)*ddif) if ddif>0 else mask
        probs = probs * mask + 1e-20
        probs = probs / probs.sum(-1, keepdim=True)
    action = action - probs.detach() + probs if hard else probs
    return action.reshape(logits.shape)

def rao_gumbel(logits, tau=1.0, mask = None, repeats=100, hard=True):
    distribution_original = masked_softmax(logits, mask=mask, dim=-1)
        
    logits_cpy = logits.detach()
    probs = masked_softmax(logits_cpy, mask)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs)  
    action = m.sample()
    action_bool = action.bool()
        
    logits_shape = logits.shape
    bs = logits_shape[0]
        
    E = torch.empty(logits_shape + (repeats,), 
                    dtype=logits.dtype, 
                    layout=logits.layout, 
                    device=logits.device,
                    memory_format=torch.legacy_contiguous_format).exponential_()
                
    Ei = E[action_bool].view(logits_shape[:-1] + (repeats,)) # rv. for the sampled location 

    wei = logits_cpy.exp()
    if mask is not None:
        wei = wei * mask.float() + 1e-20
    Z = wei.sum(dim=-1, keepdim=True) # (bs, latdim, 1)
    EiZ = (Ei/Z).unsqueeze(-2) # (bs, latdim, 1, repeats)

    new_logits = E / (wei.unsqueeze(-1)) # (bs, latdim, catdim, repeats)
    new_logits[action_bool] = 0.0
    new_logits = -(new_logits + EiZ + 1e-20).log()
    logits_diff = new_logits - logits_cpy.unsqueeze(-1)

    prob = masked_softmax((logits.unsqueeze(-1) + logits_diff)/tau, mask, dim=-2).mean(dim=-1)
    action = action - prob.detach() + prob if hard else prob
    return action.view(logits_shape), distribution_original

def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand.to(sequence_length)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    reversed_indices = reversed_indices.to(inputs).long()
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
