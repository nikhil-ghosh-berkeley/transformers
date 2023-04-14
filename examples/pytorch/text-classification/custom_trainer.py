import math
import time
from copy import deepcopy
from typing import Dict, List, Optional

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import Trainer, TrainerCallback
from transformers.custom_utils import get_param_group
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import (IterableDatasetShard,
                                           get_parameter_names)
from transformers.trainer_utils import seed_worker, speed_metrics
from transformers.utils import is_datasets_available


class CustomTrainer(Trainer):
    def get_train_dataloader(self, subset_indices=None) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if subset_indices is not None:
            train_dataset = train_dataset.select(subset_indices)
        batch_size = self._train_batch_size if subset_indices is None else self.args.per_device_eval_batch_size

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler() if subset_indices is None else None

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        eval_output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in eval_output.metrics:
            start_time += eval_output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        eval_output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=eval_output.num_samples,
                num_steps=math.ceil(eval_output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_output.metrics)
        self._memory_tracker.stop_and_update_metrics(eval_output.metrics)
        
        num_subset_samples = min(10000, len(self.train_dataset))
        subset_indices = np.random.choice(len(self.train_dataset), size=num_subset_samples)
        train_dataloader = self.get_train_dataloader(subset_indices=subset_indices)

        train_output = eval_loop(
            train_dataloader,
            description="Training Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="train",
        )

        output_metrics = {**train_output.metrics, **eval_output.metrics}
        self.log(output_metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, train_output.metrics)
        self._memory_tracker.stop_and_update_metrics(train_output.metrics)
        
        return output_metrics
        
    def create_optimizer(self):
        opt_model = self.model
        subsamp_ratio = opt_model.config.subsamp_ratio

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            param_groups = {"input": [], "output": [], "hidden": []}
            for name, _ in opt_model.named_parameters():
                param_groups[get_param_group(name)].append(name)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            lr = optimizer_kwargs["lr"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in param_groups["hidden"] and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr / subsamp_ratio
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in param_groups["output"] and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr / subsamp_ratio
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in param_groups["input"] and n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr 
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in param_groups["input"] and n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr 
                },
            ]

            
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        if control.should_evaluate:
            breakpoint()
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy