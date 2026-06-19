# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import logging
import warnings

import nncore
import torch
from deepspeed import zero
from safetensors.torch import load_model, save_file
from torch.utils.data import Sampler
from transformers import Trainer, TrainerCallback
from transformers.integrations import WandbCallback, rewrite_logs
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def gather(param):
    if hasattr(param, 'ds_id'):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def gather_lora_params(model, bias):
    assert bias in ('lora_only', 'all', 'none')

    if bias == 'lora_only':
        state_dict, maybe_lora_bias, lora_bias_names = dict(), dict(), set()
        for n, p in model.named_parameters():
            if 'modules_to_save' in n:
                state_dict[n] = p
            elif 'lora_' in n:
                state_dict[n] = p
                bias_name = n.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in n:
                maybe_lora_bias[n] = p
        for n, p in maybe_lora_bias.items():
            if n in lora_bias_names:
                state_dict[n] = p
    else:
        keys = ['lora_', 'modules_to_save', 'bias'] if bias == 'all' else ['lora_', 'modules_to_save']
        state_dict = {n: p for n, p in model.named_parameters() if any(k in n for k in keys)}

    state_dict = {n: gather(p) for n, p in state_dict.items()}
    return state_dict


def gather_non_lora_params(model):
    state_dict = {n: p for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n}
    state_dict = {n: gather(p) for n, p in state_dict.items()}
    return state_dict


class GroupSampler(Sampler):

    def __init__(self, data_types, group_size, seed):
        self.data_types = data_types
        self.group_size = group_size
        self.seed = seed

    def __len__(self):
        return len(self.data_types)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # avoid using dict or set here as they are not deterministic
        unique_types, groups = [], []
        for i, t in enumerate(self.data_types):
            if t not in unique_types:
                unique_types.append(t)
                groups.append([])
            groups[unique_types.index(t)].append(i)

        group_batches = []
        for group in groups:
            inds = [group[i] for i in torch.randperm(len(group), generator=g)]
            batches = [inds[i:i + self.group_size] for i in range(0, len(inds), self.group_size)]

            if len(batches[-1]) < self.group_size:
                batches = batches[:-1]

            group_batches += batches

        perm_group_batches = [group_batches[i] for i in torch.randperm(len(group_batches), generator=g)]
        inds = [i for batch in perm_group_batches for i in batch]

        return iter(inds)

    def set_epoch(self, epoch):
        self.epoch = epoch


class ChunkStoppingCallback(TrainerCallback):

    def __init__(self, trainer, chunk_steps):
        self.trainer = trainer
        self.chunk_steps = chunk_steps
        self.start_step = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_step is None:
            self.start_step = state.global_step - 1

        if state.global_step - self.start_step >= self.chunk_steps:
            control.should_training_stop = True
            self.trainer.finished = False

        return control


class CustomWandbCallback(WandbCallback):
    """Custom WandbCallback that uses global_step as the x-axis step."""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            # Remove data_type from wandb logs (keep it console-only)
            non_scalar_logs = {k: v for k, v in non_scalar_logs.items() if k != 'data_type'}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            # Use explicit step parameter instead of relying on internal counter
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step}, step=state.global_step)


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.chunk_steps > 0:
            self.add_callback(ChunkStoppingCallback(self, self.args.chunk_steps))

        # Replace default WandbCallback with our custom one that uses explicit step
        if self.args.report_to and 'wandb' in self.args.report_to:
            # Remove default WandbCallback (but keep CustomWandbCallback if exists)
            self.callback_handler.callbacks = [
                cb for cb in self.callback_handler.callbacks 
                if not (isinstance(cb, WandbCallback) and type(cb).__name__ == 'WandbCallback')
            ]
            # Only add custom callback if not already present
            if not any(isinstance(cb, CustomWandbCallback) for cb in self.callback_handler.callbacks):
                self.add_callback(CustomWandbCallback())
                print("Replaced default WandbCallback with CustomWandbCallback")

        self.debug_logger = self._build_debug_logger()
        self.debug_logger.debug('Custom Trainer Initialized')

    def _build_debug_logger(self):
        nncore.mkdir(self.args.output_dir)
        file_path = nncore.join(self.args.output_dir, f'debug.{self.accelerator.local_process_index}.log')
        formatter = logging.Formatter(fmt='[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(file_path, mode='a', delay=True)
        handler.setFormatter(formatter)

        logger = logging.getLogger(f'debug.{self.accelerator.local_process_index}')
        logger.setLevel(logging.DEBUG)
        logger.handlers = [handler]
        logger.propagate = False

        return logger

    def _get_train_sampler(self, *args, **kwargs):
        if self.args.group_by_data_type:
            return GroupSampler(self.train_dataset.data_types, self.args.train_batch_size * self.args.world_size,
                                self.args.seed)
        else:
            return super()._get_train_sampler(*args, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        partial_path = nncore.join(resume_from_checkpoint, 'pytorch_model.safetensors')
        if nncore.is_file(partial_path):
            load_model(model, partial_path, strict=False, device=model.device)

    def create_optimizer(self):
        if self.optimizer is None:
            grad_ps = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

            decay_ps = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_ps = [n for n in decay_ps if 'bias' not in n]

            if self.args.sam2_lr is not None:
                assert self.args.sam2_enc_lr is None and self.args.sam2_dec_lr is None
                self.args.sam2_enc_lr = self.args.sam2_lr
                self.args.sam2_dec_lr = self.args.sam2_lr

            if self.args.lora_lr is None:
                self.args.lora_lr = self.args.learning_rate

            if self.args.sam2_enc_lr is None:
                self.args.sam2_enc_lr = self.args.learning_rate

            if self.args.sam2_dec_lr is None:
                self.args.sam2_dec_lr = self.args.learning_rate

            lora_ps = [n for n, _ in grad_ps if 'lora' in n]
            sam2_enc_ps = [n for n, _ in grad_ps if 'sam2' in n and 'image_encoder' in n]
            sam2_dec_ps = [n for n, _ in grad_ps if 'sam2' in n and 'image_encoder' not in n]

            special_ps = lora_ps + sam2_enc_ps + sam2_dec_ps
            assert len(special_ps) == len(set(special_ps))

            groups = [{
                'params': [p for n, p in grad_ps if (n in decay_ps and n not in special_ps)],
                'weight_decay': self.args.weight_decay
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n not in special_ps)],
                'weight_decay': 0.0
            }, {
                'params': [p for n, p in grad_ps if (n in decay_ps and n in lora_ps)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.lora_lr
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n in lora_ps)],
                'weight_decay': 0.0,
                'lr': self.args.lora_lr
            }, {
                'params': [p for n, p in grad_ps if (n in decay_ps and n in sam2_enc_ps)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.sam2_enc_lr
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n in sam2_enc_ps)],
                'weight_decay': 0.0,
                'lr': self.args.sam2_enc_lr
            }, {
                'params': [p for n, p in grad_ps if (n in decay_ps and n in sam2_dec_ps)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.sam2_dec_lr
            }, {
                'params': [p for n, p in grad_ps if (n not in decay_ps and n in sam2_dec_ps)],
                'weight_decay': 0.0,
                'lr': self.args.sam2_dec_lr
            }]

            optim_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optim_cls(groups, **kwargs)

        return self.optimizer

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items_in_batch = super().get_batch_samples(epoch_iterator, num_batches, device)
        self.debug_logger.debug(f'[data] {[s.get('meta', 'unknown') for s in batch_samples]}')
        return batch_samples, num_items_in_batch

    def training_step(self, model, inputs, num_items_in_batch=None):
        meta = inputs.pop('meta', 'unknown')
        self.debug_logger.debug(f'[step] {meta}')
        
        # Extract data_type for logging
        if isinstance(meta, list) and len(meta) > 0:
            data_types = [m.get('data_type', 'unknown') if isinstance(m, dict) else 'unknown' for m in meta]
            # Store for logging callback
            self.current_batch_data_type = data_types[0] if len(set(data_types)) == 1 else f"mixed({len(set(data_types))})"
        else:
            self.current_batch_data_type = 'unknown'
        
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except Exception as e:
            print(f'Error batch (rank {self.accelerator.local_process_index}): {meta}')
            raise e
    
    def log(self, logs, start_time=None):
        """Override log to add data_type information"""
        if hasattr(self, 'current_batch_data_type'):
            logs['data_type'] = self.current_batch_data_type
        super().log(logs, start_time=start_time)

    def gather_and_save_model(self):
        deepspeed_zero3 = self.accelerator.deepspeed_config['zero_optimization']['stage'] == 3
        output_dir = self.args.output_dir

        if self.args.should_save:
            print(f'Saving final model to {nncore.abs_path(output_dir)}...')

        if self.processing_class is not None and self.args.should_save:
            self.processing_class.save_pretrained(output_dir)

        if self.args.save_full_model and self.args.lora_enable and deepspeed_zero3:
            warnings.warn('LoRA models cannot be saved in full mode under zero3, saving adapters instead')
            self.args.save_full_model = False

        if self.args.save_full_model:
            if self.args.lora_enable:
                self.model = self.model.merge_and_unload()

            if deepspeed_zero3 and not self.model_wrapped.zero_gather_16bit_weights_on_model_save():
                warnings.warn('Saving zero checkpoint, use zero_to_fp32.py to recover weights')
                self.model_wrapped.save_checkpoint(output_dir)
                return

            if deepspeed_zero3:
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            else:
                state_dict = self.model.state_dict()

            if self.args.should_save:
                state_dict = {k[17:] if k.startswith('base_model.model.') else k: v for k, v in state_dict.items()}
                self._save(output_dir, state_dict=state_dict)
        else:
            if self.args.lora_enable:
                state_dict = gather_lora_params(self.model, self.args.lora_bias)
                if self.args.should_save:
                    self.model.save_pretrained(output_dir, state_dict=state_dict)

            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
                self.model.generation_config.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

            state_dict = gather_non_lora_params(self.model)
            if self.args.should_save and state_dict:
                save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))

    def _save_checkpoint(self, model, trial, **kwargs):
        output_dir = self._get_output_dir(trial)
        output_dir = nncore.join(output_dir, f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')

        if self.args.should_save:
            print(f'Saving checkpoint to {nncore.abs_path(output_dir)}...')

        super()._save_checkpoint(model, trial, **kwargs)

        if self.processing_class is not None and self.args.should_save:
            self.processing_class.save_pretrained(output_dir)

        if self.args.lora_enable:
            state_dict = gather_non_lora_params(self.model)
            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
                save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))
