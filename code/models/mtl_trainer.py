import collections
import transformers

import numpy as np
from packaging import version
import json
import sys
import os
import pickle
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from transformers import BertTokenizerFast, TrainingArguments, BertConfig, Trainer, AdamW, logging
from transformers.trainer import get_parameter_names, Adafactor
from transformers.training_args import ParallelMode
from transformers.trainer_pt_utils import DistributedSamplerWithLoop, IterableDatasetShard, ShardSampler, \
    LengthGroupedSampler, DistributedLengthGroupedSampler
from transformers.file_utils import is_datasets_available
import datasets

from .mtl_models import DataLoaderWithTaskname, MultitaskDataloader, MultitaskModel
from typing import List, Optional

logger = logging.get_logger(__name__)
_is_torch_generator_available = False

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True


class MultitaskTrainer(transformers.Trainer):

    def __init__(self, collator_dict, batch_size_dict, params, compute_metrics_tasks, best_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collator_dict = collator_dict
        self.batch_size_dict = batch_size_dict
        self.params = params
        self.compute_metrics_tasks = compute_metrics_tasks
        if isinstance(best_scores, str):
            self.best_scores_path = best_scores
            with open(best_scores, 'rb') as f:
                self.best_scores = pickle.load(f)
        else:
            self.best_scores_path = os.path.join(self.args.output_dir, 'best_score.pkl')
            self.best_scores = best_scores
            if self.args.local_rank in [-1, 0]:
                with open(self.best_scores_path, 'wb') as f:
                    pickle.dump(self.best_scores, f)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n in decay_parameters and n in self.params],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n in decay_parameters and n not in self.params],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n not in decay_parameters and n in self.params],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n not in decay_parameters and n not in self.params],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(train_dataset, generator=generator)
                return RandomSampler(train_dataset)
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

    def get_single_train_dataloader(self, task_name, train_dataset, collator_dict, batch_size_dict):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=batch_size_dict[task_name] * max(1, self.args.n_gpu),
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            data_loader = DataLoaderWithTaskname(
                task_name=task_name,
                data_loader=DataLoader(
                    train_dataset,
                    batch_size=batch_size_dict[task_name] * max(1, self.args.n_gpu),
                    collate_fn=collator_dict[task_name],
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                ),
            )
            return data_loader

        train_sampler = self._get_train_sampler(train_dataset)
        data_loader = DataLoaderWithTaskname(
                task_name=task_name,
                data_loader=DataLoader(
                    train_dataset,
                    batch_size=batch_size_dict[task_name] * max(1, self.args.n_gpu),
                    sampler=train_sampler,
                    collate_fn=collator_dict[task_name],
                    drop_last=self.args.dataloader_drop_last,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                ),
            )

        return data_loader

    def get_single_eval_dataloader(self, task_name, eval_dataset, collator_dict, batch_size_dict):
        if self.eval_dataset is None:
            raise ValueError("Evaluation: evaluation requires a train_dataset.")
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=batch_size_dict[task_name] * max(1, self.args.n_gpu),
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            data_loader = DataLoaderWithTaskname(
                task_name=task_name,
                data_loader=DataLoader(
                    eval_dataset,
                    batch_size=batch_size_dict[task_name] * max(1, self.args.n_gpu),
                    collate_fn=collator_dict[task_name],
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                ),
            )
            return data_loader

        eval_sampler = self._get_eval_sampler(eval_dataset)
        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=batch_size_dict[task_name] * max(1, self.args.n_gpu),
                sampler=eval_sampler,
                collate_fn=collator_dict[task_name],
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        collator_dict = self.collator_dict
        batch_size_dict = self.batch_size_dict
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset, collator_dict=collator_dict,
                                                        batch_size_dict=batch_size_dict)
            for task_name, task_dataset in self.train_dataset.items()
        })

    def get_eval_dataloader(self, eval_dataset):
        collator_dict = self.collator_dict
        batch_size_dict = self.batch_size_dict
        return {
            task_name: self.get_single_eval_dataloader(task_name, task_dataset, collator_dict=collator_dict,
                                                       batch_size_dict=batch_size_dict)
            for task_name, task_dataset in self.eval_dataset.items()
        }

    def save_best_score(self):
        with open(self.best_scores_path, 'wb') as f:
            pickle.dump(self.best_scores, f)

    def load_best_score(self):
        with open(self.best_scores_path, 'rb') as f:
            self.best_scores = pickle.load(f)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        # memory metrics - must set up as early as possible
        results = {}
        self._memory_tracker.start()
        self.load_best_score()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        for task, dataloader in eval_dataloader.items():
            self.compute_metrics = self.compute_metrics_tasks[task]
            output = eval_loop(
                dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            tasks_metric = {"eval_" + task + "_" + k: v for k, v in output.metrics.items()}
            main_metric = None
            for k, v in tasks_metric.items():
                if 'loss' not in k:
                    main_metric = v
            if main_metric > self.best_scores[task]:
                checkpoint_folder = f"{task}/checkpoint-best"
                save_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.save_model(output_dir=save_dir)
                self.best_scores[task] = main_metric
                if self.args.local_rank in [-1, 0]:
                    with open(self.best_scores_path, 'wb') as f:
                        pickle.dump(self.best_scores, f)
            for key in sorted(tasks_metric.keys()):
                print(key, '=', tasks_metric[key])
                logger.info(f"  {key} = {tasks_metric[key]}")
            results.update(tasks_metric)

        # Computes the average metrics across all the tasks without their corresponding losses.
        metrics = [results[key] for key in results.keys() if "loss" not in key]
        results['eval_average_metrics'] = np.mean(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, results)
        self.save_best_score()

        return results