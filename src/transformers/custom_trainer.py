from functools import reduce

import torch.nn as nn

from transformers import Seq2SeqTrainer, Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging
from peft.tuners import lora

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.get_logger(__name__)

def get_module(name, opt_model):
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module


def _create_optimizer(opt_model, args):
    """
    Setup the optimizer.
    """
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }
    breakpoint()
    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue
        
        module = get_module(name, opt_model)
        if isinstance(module, nn.Embedding) or isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_A" in name:
            param_groups["groupA"][name] = param
        elif "lora_B" in name:
            param_groups["groupB"][name] = param
        elif "score" in name or "classifier" in name or "lm_head" in name:
            if isinstance(module, nn.Linear):
                param_groups["groupA"][name] = param
        elif param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    logger.info(assigned_param_groups)

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    lr = args.learning_rate
    lr_ratio = args.lr_ratio
    lr_embedding = args.lr_embedding

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": args.weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": args.weight_decay,
            "lr": lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": args.weight_decay,
            "lr": lr * lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")
    breakpoint()
    return optimizer


class CustomTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.
        Note: sagemaker and fairscale not tested in this override.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            self.optimizer = _create_optimizer(opt_model, self.args)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        """
        Setup the optimizer.
        Note: sagemaker and fairscale not tested in this override.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            self.optimizer = _create_optimizer(opt_model, self.args)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
