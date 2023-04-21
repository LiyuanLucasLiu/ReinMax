# The code has been modified from https://github.com/chijames/GST

import torch.nn.functional as F

from .basic import rao_gumbel, gst_mover, exact, \
    st, reinmax, reinmax_mean_baseline 
    
simple_method_mapping = {
    'reinmax': reinmax,
    'st': st, 
    'exact': exact,
    'reinmax_mean_baseline': reinmax_mean_baseline,
}
repeat_method_mapping = {
    'rao': ('rao_gumbel', rao_gumbel, int),
}
def categorical_repara(logits, temp, method = 'gumbel'):        
    if method == 'gumbel':
        qy = F.softmax(logits, dim=-1)
        return F.gumbel_softmax(logits, tau=temp, hard=True), qy
    
    elif method in simple_method_mapping:
        sample, qy = simple_method_mapping[method](logits, tau=temp)
        return sample, qy

    elif method.startswith('gst-'):
        try:
            gap = float(method[4:])
            if gap<0.0: gap = 'p'
        except:
            gap = 'p'
        ret = gst_mover(logits, temp, gap=gap)
        qy = F.softmax(logits, dim=-1)
        return ret, qy
    
    elif len(method) > 3 and method[:3] in repeat_method_mapping:
        name, func, typ = repeat_method_mapping[method[:3]]
        assert method.startswith(name)
        try:
            repeats=typ(method[len(name) + 1:])
        except:
            repeats=100
        sample, qy = func(logits, tau=temp, repeats=repeats)
        return sample, qy
        
    else:
        print(method + ' not supported')
        exit()
