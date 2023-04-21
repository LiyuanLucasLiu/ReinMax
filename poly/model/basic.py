# The code has been modified from https://github.com/chijames/GST

import torch

def st(logits, tau):
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
        
def exact(logits, tau=1.0):
    m = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
    action = m.sample()
    prob = logits.softmax(dim=-1)
    return action, prob

class ReinmaxMean_KERNEL(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        logit: torch.Tensor, 
        p: torch.Tensor,
        p_0: torch.Tensor,
        sample: torch.Tensor,
    ):
        assert logit.dim() == 2
        ctx.save_for_backward(p, p_0, sample)
        return torch.zeros_like(p, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)

    @staticmethod
    def backward(ctx, grad_at_output):
        p, p_0, sample = ctx.saved_tensors
        one_hot_sample = torch.zeros_like(p, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)
        grad_fo_0 = grad_at_output.gather(dim=-1, index=sample) - grad_at_output.mean(dim=-1, keepdim=True)
        grad_fo_1 = one_hot_sample - p
        grad_fo = grad_fo_0 * grad_fo_1
        
        grad_st_0 = grad_at_output * p
        grad_st_1 = grad_st_0 * one_hot_sample - grad_st_0.sum(dim=-1, keepdim=True) * p
        N = p_0.size(-1)
        grad_st = grad_st_1 / (N * p_0.detach().gather(dim=-1, index=sample) + 1e-12)
        
        grad_at_input = .5 * (grad_fo + grad_st)
        return grad_at_input, None, None, None

def reinmax_mean_baseline(logits, tau=1.0):    
    shape, _ = logits.size(), logits.dtype
    logits = logits.view(-1, shape[-1])
    y_soft = logits.softmax(dim=-1)
    
    sample = torch.multinomial(
        y_soft,
        num_samples=1,
        replacement=True,
    )
    
    y_soft_tau = (logits/tau).softmax
    
    onehot_sample = ReinmaxMean_KERNEL.apply(logits, y_soft, y_soft_tau, sample)
    return onehot_sample.view(shape), y_soft.view(shape)

def gst_mover(logits, tau=1.0, hard=True, gap=1.0, detach=True):
    logits_cpy = logits.detach() if detach else logits
    probs = logits_cpy.softmax(dim=-1)
        
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
    probs = (logits / tau).softmax(dim=-1)
    action = action - probs.detach() + probs if hard else probs
    return action.reshape(logits.shape)

def rao_gumbel(logits, tau=1.0, repeats=100, hard=True):
    distribution_original = logits.softmax(dim=-1)
        
    logits_cpy = logits.detach()
    probs = logits_cpy.softmax(dim=-1)
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
    Z = wei.sum(dim=-1, keepdim=True) # (bs, latdim, 1)
    EiZ = (Ei/Z).unsqueeze(-2) # (bs, latdim, 1, repeats)

    new_logits = E / (wei.unsqueeze(-1)) # (bs, latdim, catdim, repeats)
    new_logits[action_bool] = 0.0
    new_logits = -(new_logits + EiZ + 1e-20).log()
    logits_diff = new_logits - logits_cpy.unsqueeze(-1)

    prob = ((logits.unsqueeze(-1) + logits_diff)/tau).softmax(dim=-2).mean(dim=-1)
    action = action - prob.detach() + prob if hard else prob
    return action.view(logits_shape), distribution_original
