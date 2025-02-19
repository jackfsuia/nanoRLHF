# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import psutil
import textwrap
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
import logging
from memory_profiler import profile
import torch.nn.utils.rnn as rnn_utils

from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.utils import generate_model_card,generate
import pickle
import random
random.seed(42)

if is_wandb_available():
    import wandb

TRAINER_STATE_NAME = "trainer_state.json"

INVALID_LOGPROB = 1.0
def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=False,
        output_hidden_states=False,
        use_cache=False
    )
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy,) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **kwargs):
        return self.policy(**kwargs)
def vllm_generate(grpo_sample_N,model,tokenizer,prompts,temperature,max_tokens):
    device = model.policy.device
    model.to('cpu')
    torch.cuda.empty_cache()
    policy = model.policy
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, n=grpo_sample_N, max_tokens=max_tokens, logprobs =1,detokenize=False,ignore_eos=False, seed=random.randint(1, 5000))
    save_model_path = '/data/temp_vllm_model'
    shutil.rmtree(save_model_path,ignore_errors=True)

    if policy.peft_type:
        save_adapter_path = save_model_path+'/adapter'
        policy.save_pretrained(save_adapter_path)
        temp_policy = AutoModelForCausalLM.from_pretrained(policy.name_or_path, trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cpu')
        temp_policy = PeftModel.from_pretrained(temp_policy, save_adapter_path)
        temp_policy.merge_and_unload()
        save_full_model_path = save_model_path + '/full_model'
        temp_policy.base_model.model.save_pretrained(save_full_model_path)
        del temp_policy
        tokenizer.save_pretrained(save_full_model_path)
        llm = LLM(model=save_full_model_path)
        outputs_ALL = llm.generate(prompts, sampling_params)
    else:
        save_model_path = '/data/temp_vllm_model'
        shutil.rmtree(save_model_path,ignore_errors=True)
        policy.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
        llm = LLM(model=save_model_path)
        outputs_ALL = llm.generate(prompts, sampling_params)
    shutil.rmtree(save_model_path)

    response =[]
    for o in outputs_ALL:
        for o_i in o.outputs:
            paded_token_ids =o_i.token_ids + (tokenizer.pad_token_id,)*(max_tokens-len(o_i.token_ids))
            tokens = torch.tensor(paded_token_ids)
            response+=[tokens]

    response = torch.stack(response, dim=0)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    model.to(device)

    return response

def state_to_device(state, device):
    for _, v in state.items():
        if isinstance(v, dict):
            for k in v.keys():
                v[k] = v[k].to(device)

class GRPOTrainer(Trainer): # sparse realization
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        config: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
        reward_func=None,
        accuracy_func =None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47
        self.reward_func=reward_func
        self.accuracy_func=accuracy_func
        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        # for minibatch inner dynamic batching
        accelerator = Accelerator(gradient_accumulation_steps = 1)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = processing_class.eos_token_id
        self.model = PolicyAndValueWrapper(policy)
        self.model.config = policy.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        #########
        ### setup dataloader
        #########

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(self.processing_class),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
            worker_init_fn=seed_worker,
            generator=g,
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader, device_placement=[False, False, False])
        torch.manual_seed(self.local_seed)  # reset the local seed again


    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"


        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                    f"which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. "
                    f"Please ensure that the `compute_metrics` function returns a dictionary that includes '{metric_to_check}' or "
                    f"consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                # Since the metric is one step behind, change to save the previous checkpoint.
                if metric_to_check.endswith('_old'):
                    best_checkpoint_path = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{max(self.state.global_step-self.args.save_steps,1)}")
                    if os.path.exists(best_checkpoint_path):
                        self.state.best_model_checkpoint = best_checkpoint_path
                    else:
                        self.state.best_model_checkpoint = output_dir
                else:
                    self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)  

# Dynamic batching/Bucket Batching, all about saving time
    def _create_batches(self, lengths, max_batch_memory_size):
        sorted_indices = torch.argsort(lengths, descending=False)
        batches_indices = []
        current_batch_indices = []
        current_batch_memory = 0
        cur_len = 0
        for idx in sorted_indices:

            sample_len = lengths[idx]
            future_batch_memory = max(cur_len, sample_len)*(len(current_batch_indices)+1)

            if future_batch_memory > max_batch_memory_size and len(current_batch_indices) > 0:
                # If the current batch is full, save the current batch and start a new batch.
                batches_indices.append(current_batch_indices)
                current_batch_indices = []
                current_batch_memory = 0
                cur_len = 0
            current_batch_indices.append(idx.item())
            current_batch_memory += sample_len
            cur_len  = max(cur_len, sample_len)

        # save last batch
        if len(current_batch_indices) > 0:
            batches_indices.append(current_batch_indices)

        return batches_indices

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        processing_class = self.processing_class
        dataloader = self.dataloader
        main_device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=args.temperature,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches)
        approxkl_stats = torch.zeros(stats_shape, device=main_device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=main_device)
        pg_loss_stats = torch.zeros(stats_shape, device=main_device)
        entropy_stats = torch.zeros(stats_shape, device=main_device)
        ratio_stats = torch.zeros(stats_shape, device=main_device)
        refkl_stats = torch.zeros(stats_shape, device=main_device)

        
        if args.accuracy_before_train:
            accelerator.print("===initial evaluation===")
            initial_accuracy = self.accuracy_func(model,self.args)
        else:
            initial_accuracy = -1

        model.train()

        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.gradient_checkpointing:
            self.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model.to(main_device)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            start_time = time.time()
            state_to_device(optimizer.optimizer.state,'cpu')
            torch.cuda.empty_cache()

            with torch.no_grad():

                queries = data["input_ids"].to(main_device)     
                batch_size = queries.shape[0]


                quesiton_string = self.tokenizer.batch_decode(queries)
                quesiton_string = [q.replace('[PAD]','') for q in quesiton_string]

                # each quesiton_string with args.grpo_sample_N samples
                responses = vllm_generate(args.grpo_sample_N,model,self.tokenizer, quesiton_string, args.temperature, max_tokens=generation_config.max_new_tokens).to(main_device)


                responses_decoded = self.tokenizer.batch_decode(responses)

                quesiton_string_N = [item_s for item_s in quesiton_string for _ in range(args.grpo_sample_N)]

                scores = self.reward_func([i+j for i, j in zip(quesiton_string_N, responses_decoded)],responses,self.tokenizer).to(main_device)

                # save  for logger
                log_scores = scores.clone()
                rlhf_reward = scores.mean()

                # GRPO. We randomly keep batch_size samples, abandon the rest batch_size*(grpo_sample_N-1) samples after getting the baseline to save for&backward time
                random_indices = torch.randint(0, args.grpo_sample_N, (batch_size,))
                reshaped_scores = scores.view(-1,args.grpo_sample_N)

                # score = score - score_baseline, this is actually advantage.
                reshaped_scores = (reshaped_scores - reshaped_scores.mean(dim=1, keepdim=True))/reshaped_scores.std(dim=1, keepdim=True)
                reshaped_scores = reshaped_scores[torch.arange(batch_size), random_indices]
                scores = reshaped_scores.view(-1)

                # filter out the 0/0 error
                scores[torch.isnan(scores)] = 0

                reshaped_responses = responses.view(batch_size,args.grpo_sample_N,-1)
                responses = reshaped_responses[torch.arange(batch_size), random_indices]

                # save for logger
                log_scores =  log_scores.view(-1,args.grpo_sample_N)
                log_scores = log_scores[torch.arange(batch_size), random_indices]
                log_scores = log_scores.view(-1)
                responses_decoded = [responses_decoded[i * args.grpo_sample_N + idx] for i, idx in enumerate(random_indices)]
                responses_decoded = [r.replace('[PAD]','') for r in responses_decoded]


                gc.collect()
                log_responses_length = first_true_indices(responses == processing_class.pad_token_id)
                
                # Here we use a "sparse" implementation to save time, which is, we abandon the query-reponses samples whose advantage is 0. We assume and observed that this is ok.-------

                # 1. filter out scores=0 samples
                non_zero_sample_idx = torch.where(scores != 0)[0]
                scores = scores[non_zero_sample_idx]
                queries = queries[non_zero_sample_idx,:]
                responses = responses[non_zero_sample_idx,:]
                
                # 2. remove redundant padding-----------------------------
                # partial depadding
                q_paddding_len = first_true_indices(queries != processing_class.pad_token_id)
                q_start_idx = q_paddding_len.min()
                queries = queries[:,q_start_idx:]
                
                 # partial depadding
                responses_length = first_true_indices(responses == processing_class.pad_token_id)
                max_response_length = responses_length.max()
                responses = responses[:, :max_response_length]
                
                query_responses = torch.cat((queries, responses), dim=1)
                
                context_length = queries.shape[1]
                query_responses_length = context_length + responses_length
                #--------------------------------------------------------------------------------------------------
                ref_policy.to(main_device)

                # create dynamic bucket batch, all about saving time
                query_response_batches_idxs  = self._create_batches(query_responses_length, 22*2316)# model-wise memory magic numer:22*2316
                
                query_responses_batches = [query_responses[idxs][:,:query_responses_length[idxs].max()]  for idxs in query_response_batches_idxs]
                
                logprobs = []
                ref_logprobs = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    unwrapped_model.eval()

                    for query_response, rixs in tqdm(zip(query_responses_batches, query_response_batches_idxs),desc='forward prob data'):
                        response = query_response[:, context_length:]
                        logprob = forward(unwrapped_model.policy, query_response, processing_class.pad_token_id)
                        logprob = logprob[0][:, context_length - 1 : -1].clone()
                        logprob /= args.temperature
                        logprob = F.log_softmax(logprob, dim=-1)
                        logprob = torch.gather(logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        ref_logprob = forward(ref_policy, query_response, processing_class.pad_token_id)
                        ref_logprob = ref_logprob[0][:, context_length - 1 : -1].clone()
                        ref_logprob /= args.temperature
                        ref_logprob = F.log_softmax(ref_logprob, dim=-1)
                        ref_logprob = torch.gather(ref_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        
                        
                        logprob = torch.nn.functional.pad(logprob, (0, max_response_length - logprob.shape[1]), mode='constant', value=INVALID_LOGPROB)
                        ref_logprob = torch.nn.functional.pad(ref_logprob, (0, max_response_length - ref_logprob.shape[1]), mode='constant', value=INVALID_LOGPROB)

                        logprobs.append(logprob)
                        ref_logprobs.append(ref_logprob)
                        
                        
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)

                # adjust the orders accordingly
                qrb_idx =[idxt for subidce in query_response_batches_idxs for idxt in subidce]
                queries = queries[qrb_idx,:]
                responses = responses[qrb_idx,:]
                responses_length = responses_length[qrb_idx]
                query_responses = query_responses[qrb_idx,:]
                query_responses_length = query_responses_length[qrb_idx]
                scores = scores[qrb_idx]
                
                before_eos_lengths = responses_length - 1
                
                del (ref_logprob, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()
                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > before_eos_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = before_eos_lengths + 1

                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))

                # 4. compute rewards

                rewards = torch.zeros_like(ref_logprobs)
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, before_eos_lengths)

                # 4.5 shape reward
                normal_scores = scores
                rewards[[actual_start, actual_end]] += normal_scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=True)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages
                lastgaelam = 0
                advantages_reversed = []
                for t in reversed(range(max_response_length)):
                    lastgaelam = rewards[:, t] + lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                if args.advantage_whiten:
                    advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()
            ref_policy.to("cpu")
            torch.cuda.empty_cache()

            state_to_device(optimizer.optimizer.state, main_device)
            torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            model.train()
            
            modified_local_batch_size = query_responses.shape[0]
            
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(modified_local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in tqdm(range(0, modified_local_batch_size, args.local_mini_batch_size),desc='forward and backward'):
                    # sparse realization
                    with torch.no_grad():
                        
                        mini_batch_end = mini_batch_start + args.local_mini_batch_size
                        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                        # since we delete the score 0 samples, batch size might have changed
                        modified_mini_batch_size = len(mini_batch_inds)
                        # create dynamic bucket batch, all about saving time
                        query_responses_response_idxs = self._create_batches(query_responses_length[mini_batch_inds], max_batch_memory_size=4*2316)                    

                        responses_lengths_temp = responses_length[mini_batch_inds]
                        responses_max_len_batches = [responses_lengths_temp[idx].max() for idx in query_responses_response_idxs]

                        query_responses_length_temp = query_responses_length[mini_batch_inds]
                        query_responses_max_length_batches = [query_responses_length_temp[idx].max() for idx in query_responses_response_idxs]

                        # depadding in bucket to save time
                        query_responses_temp = query_responses[mini_batch_inds]
                        query_responses_batches = [query_responses_temp[idx][:,:max_len] for idx, max_len in zip(query_responses_response_idxs, query_responses_max_length_batches)]

                        # depadding in bucket to save time                   
                        responses_temp = responses[mini_batch_inds]
                        responses_batches = [responses_temp[idx][:,:max_len] for idx, max_len in zip(query_responses_response_idxs, responses_max_len_batches)]

                        # depadding in bucket to save time                   
                        logprobs_temp = logprobs[mini_batch_inds] 
                        logprobs_batches = [logprobs_temp[idx][:,:max_len] for idx, max_len in zip(query_responses_response_idxs, responses_max_len_batches)]

                        # depadding in bucket to save time                   
                        advantages_temp = advantages[mini_batch_inds]
                        advantages_batches = [advantages_temp[idx][:,:max_len] for idx, max_len in zip(query_responses_response_idxs, responses_max_len_batches)]

                         # depadding in bucket to save time                   
                        ref_logprobs_temp = ref_logprobs[mini_batch_inds]
                        ref_logprobs_batches = [ref_logprobs_temp[idx][:,:max_len] for idx, max_len in zip(query_responses_response_idxs, responses_max_len_batches)]
                        # depadding in bucket to save time                   
                        padding_mask_temp = padding_mask[mini_batch_inds]
                        padding_mask_batches = [padding_mask_temp[idx][:,:max_len] for idx, max_len in zip(query_responses_response_idxs, responses_max_len_batches)]

                    for (
                        mb_advantage,
                        mb_responses,
                        mb_query_responses,
                        mb_logprobs,
                        mb_ref_logprobs,
                        padmask,
                    ) in zip(
                        advantages_batches,
                        responses_batches,
                        query_responses_batches,
                        logprobs_batches,
                        ref_logprobs_batches,
                        padding_mask_batches,
                    ):

                        output= forward(model, mb_query_responses, processing_class.pad_token_id)
                        logits = output[0][:, context_length - 1 : -1]
                        logits /= args.temperature
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)

                        new_logprobs = torch.masked_fill(new_logprobs, padmask, INVALID_LOGPROB)

                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)

                        # GRPO loss
                        kl = new_logprobs - mb_ref_logprobs
                        kl_statics = kl.detach()
                        kl = args.kl_coef * (torch.exp(-kl) +kl -1)
                        pg_loss_max = torch.max(pg_losses, pg_losses2) + kl
                        pg_loss = masked_mean(pg_loss_max, ~padmask)
                        loss = pg_loss
                    
                        del new_all_logprobs
                        
                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padmask
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            del (logits, prob_dist, output)
                            torch.cuda.empty_cache()
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx] += mb_query_responses.shape[0]*approxkl/modified_mini_batch_size
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx] +=  mb_query_responses.shape[0]*pg_clipfrac/modified_mini_batch_size
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx] +=  mb_query_responses.shape[0]*pg_loss/modified_mini_batch_size
                            entropy_stats[ppo_epoch_idx, minibatch_idx] +=  mb_query_responses.shape[0]*entropy.mean()/modified_mini_batch_size
                            ratio_stats[ppo_epoch_idx, minibatch_idx ] +=  mb_query_responses.shape[0]*ratio.mean()/modified_mini_batch_size
                            refkl_stats[ppo_epoch_idx, minibatch_idx ]+=  mb_query_responses.shape[0]*kl_statics.mean()/modified_mini_batch_size
                            torch.cuda.empty_cache()
                            
                        loss = mb_query_responses.shape[0]*loss/modified_mini_batch_size # Make sure the scale is correct.
                        loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    minibatch_idx += 1

                    del (
                        new_logprobs, 
                        logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    torch.cuda.empty_cache()

            with torch.no_grad():
                mean_entropy = (-logprobs).sum(1).mean()
                ### report case score
                table = defaultdict(list)
                table["query"].extend(quesiton_string)
                table["response"].extend(responses_decoded)
                table["score"].extend(log_scores.float().cpu().numpy())
                table["rlhf_score"].extend(log_scores.float().cpu().numpy())

                df = pd.DataFrame(table)
                if self.accelerator.is_main_process:
                    print_rich_table(df.iloc[0 : 0 + 5])
                    if "wandb" in args.report_to:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"completions": wandb.Table(dataframe=df)})

                print(f"-------------------{(time.time() - start_time) / args.batch_size} s/episode")
                metrics = {}
                metrics['initial_accuracy'] = initial_accuracy
                state_to_device(optimizer.optimizer.state,'cpu')
                if self.state.global_step % 10 == 0:
                    metrics['eval_accuracy_new'] = self.accuracy_func(model,self.args)
                metrics["eval_response_length"] = log_responses_length.float().mean().item()
                metrics["objective/kl_old"] = self.accelerator.gather(refkl_stats).mean().item()
                metrics["objective/entropy_old"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward_old"] = 0
                metrics["eval_objective/rlhf_reward_old"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["eval_objective/scores_old"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["policy/approxkl_avg_new"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg_new"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg_new"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["policy/entropy_avg_new"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio_new"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var_new"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens_old"] = (log_responses_length < args.response_length).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)
            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_entropy, scores, metrics
            torch.cuda.empty_cache()
            gc.collect()
            del (
                query_responses,
                responses,
                logprobs,
                ref_logprobs,
                before_eos_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
            )
            torch.cuda.empty_cache()
        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
