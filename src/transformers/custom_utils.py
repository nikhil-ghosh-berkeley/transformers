import torch
from typing import Tuple, List
import numpy as np

class SelectorGenerator:
    def __init__(self) -> None:
        self.selector_dict = dict()
    
    def generate(self, base_dims: Tuple, sub_dims: Tuple, random: bool = False) -> List[np.ndarray]:
        n = len(base_dims)
        assert len(sub_dims) == n
        selectors = []
        for i in range(n):
            dim_pair = (base_dims[i], sub_dims[i])
            if not random and dim_pair in self.selector_dict:
                selector = self.selector_dict[dim_pair]
            else:
                selector = np.sort(np.random.choice(base_dims[i], size=sub_dims[i], replace=False))
                if not random:
                    self.selector_dict[dim_pair] = selector
            selectors.append(selector)
        return selectors

def get_param_group(name):
    if "ln" in name:
        return "input"
    elif name.startswith("transformer.h") and name.endswith("weight"):
        return "hidden"
    elif name == "score.weight":
        return "output"
    else:
        return "input"

def compute_grad_norm(parameters, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == float("inf"):
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm.item()