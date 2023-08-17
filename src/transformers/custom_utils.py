import torch
from typing import Tuple, List
import numpy as np
import os
import json
from .utils import logging

logger = logging.get_logger(__name__)

# takes string of floats and maps to tuple of floats
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(float, strings.split(","))
    return tuple(mapped_int)

class SelectorGenerator:
    def __init__(self, subsamp_ratio: float) -> None:
        self.selector_dict = dict()
        self.subsamp_ratio = subsamp_ratio
        self.locked = False

    def load(self, save_dir: str):
        save_path = os.path.join(save_dir, f"subsamp_selector_{self.subsamp_ratio}.json")
        assert os.path.exists(save_path)
        logger.info("Loading previous subsample selection.")
        with open(save_path, 'r') as f:
            self.selector_dict = json.load(f)
        self.locked = True
    
    def save(self, save_dir: str):
        save_path = os.path.join(save_dir, f"subsamp_selector_{self.subsamp_ratio}.json")
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(save_path):
            logger.info("saved subsample selector exists skipping save")
        else:
            with open(save_path, 'w') as f:
                logger.info(f"Saving subsample selector to {save_path}")
                json.dump(self.selector_dict, f)

    def generate(
        self, base_dims: Tuple, sub_dims: Tuple
    ) -> List[np.ndarray]:
        n = len(base_dims)
        assert len(sub_dims) == n
        selectors = []
        for i in range(n):
            dim_pair = f"({base_dims[i]}, {sub_dims[i]})"
            if self.locked:
                assert dim_pair in self.selector_dict
            if dim_pair in self.selector_dict:
                selector = self.selector_dict[dim_pair]
            else:
                selector = np.sort(
                    np.random.choice(base_dims[i], size=sub_dims[i], replace=False)
                )
                selector = selector.tolist()
                if not self.locked:
                    self.selector_dict[dim_pair] = selector
            selectors.append(selector)
        return selectors

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