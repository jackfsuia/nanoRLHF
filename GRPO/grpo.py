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
import numpy as np
import shutil
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
)
from vllm import LLM, SamplingParams

from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    EarlyStoppingCallback
)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import ModelConfig, PPOConfig, PPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
import pickle
from accelerate import Accelerator
from datasets import Dataset
import json,os
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
import time
from dataclasses import dataclass,field
from typing import List, Literal, Optional
from accelerate.utils import broadcast, gather_object
from grpo_trainer import GRPOTrainer
from memory_profiler import profile
import psutil
from multiprocessing import Process, set_start_method, Queue,get_start_method
import sys
import random
random.seed(0)
torch.manual_seed(42)

# ALL parameters -----------------------------------------------------------------------------------------------------------

base_model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
project_name = "rlhf"
os.environ["WANDB_PROJECT"] = project_name

@dataclass
class GRPOConfig(PPOConfig):
    save_value_model: bool = field(default=True)
    early_stopping_patience: int = field(default=1000000)
    use_lora :bool = field(default=True)
    lora_r:int = field(default=64)
    lora_alpha :int = field(default=16)
    lora_dropout :float = field(default= 0.0)
    lora_target_modules :list[str] =field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj","down_proj"])
    modules_to_save :list[str] =field(default_factory=lambda: ["embed_tokens", "lm_head","score"])
    
    reward_batch_size :int = field(default=16)
    
    lora_bias:str = field(default="none")

    train_dataset_name :str = field(default="Anthropic/hh-rlhf")
    train_dataset_split:str = field(default="train[:100%]")

    advantage_whiten :bool = field(default= False)# this is default to be False since GRPO has provided a baseline

    grpo_sample_N:int =  field(default=4)

training_args = GRPOConfig(
    exp_name = 'grpo-v1',
    reward_model_path = 'OpenAssistant/reward-model-deberta-v3-large-v2',
    whiten_rewards=False,
    kl_coef=0.01,#0.01 
    cliprange = 0.2,#0.2
    cliprange_value = 0.01,
    lam=0.95,#0.95 
    temperature = 0.9,

    learning_rate = 6e-6,  #6e-6
    warmup_steps = 0, #4
    lr_scheduler_type = "cosine_with_min_lr",
    lr_scheduler_kwargs ={'min_lr_rate':0.1},

    response_length=1500,
    
    ## mini_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    ## batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_mini_batches
    per_device_train_batch_size = 4,#4
    gradient_accumulation_steps = 8,#8
    num_mini_batches = 16,#16
    num_ppo_epochs = 1,

    total_episodes=250000,#100000
    local_rollout_forward_batch_size = 16,
    bf16 = True,
    gradient_checkpointing = True,
    missing_eos_penalty=None,
    report_to="wandb",

    sft_model_path=base_model_name_or_path,

    eval_strategy= "steps",
    eval_steps = 1,
    save_strategy ="steps",
    save_steps =1 ,
    save_total_limit= 8 ,
    log_level= 'info',
    output_dir=f"{base_model_name_or_path}/{project_name}",
    logging_dir =f"{base_model_name_or_path}/{project_name}/logs" ,
    logging_strategy ="steps" ,
    logging_steps =1,
    metric_for_best_model= "eval_objective/rlhf_reward_old",
    greater_is_better = True,
    load_best_model_at_end =True,
    stop_token = "eos",
    dataset_num_proc = 6
)

# load reward model to cpu, define reward function ------------------------------------

reward_model, rm_tokenizer = AutoModelForSequenceClassification.from_pretrained(training_args.reward_model_path,device_map='cpu').eval(), AutoTokenizer.from_pretrained(training_args.reward_model_path)


def reward_func(pmt_and_responses, eos_token):
    
    reward_model.to('cuda')
    
    episode_num = len(pmt_and_responses)

    rewards  = torch.zeros(episode_num)
    
    questions = []
    responses=[]
    
    for pmt_and_response in pmt_and_responses:
        # get the question and response
        question_idx = pmt_and_response.find("user\n") + len("user\n")
        question_end_idx = pmt_and_response.find("<|im_end|>", question_idx)
        question = pmt_and_response[question_idx:question_end_idx]
        response_idx = pmt_and_response.find('<|im_start|>assistant\n')+len('<|im_start|>assistant\n')
        endix = pmt_and_response.find(eos_token, response_idx)
        if endix==-1:
            response = pmt_and_response[response_idx:]
        else:
            response = pmt_and_response[response_idx:endix]
        
        questions.append(question)
        responses.append(response)
        
    # reward the question and response, per https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2
    for i in tqdm(range(0, episode_num, training_args.reward_batch_size), desc="Rewarding"):        
        inputs = rm_tokenizer(questions[i:i+training_args.reward_batch_size], responses[i:i+training_args.reward_batch_size], return_tensors='pt',padding=True)
        inputs = {key: value.to(reward_model.device) for key, value in inputs.items()}
        rewards[i:i+training_args.reward_batch_size] = reward_model(**inputs).logits.cpu().detach().squeeze()

        
    reward_model.to('cpu')
    torch.cuda.empty_cache()
    
    return rewards


if __name__ == "__main__":

    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    
# load policy, reference model to cpu ------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        padding_side="left",
        trust_remote_code=False,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",
    ).eval()
    
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2"
    )
    
    if training_args.use_lora:
  
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=training_args.modules_to_save  # This argument serves for adding new tokens.
        )
       
        policy = get_peft_model(policy, lora_config)

        policy.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            policy.enable_input_require_grads()

# load dataset ------------------------------------------

    train_dataset = load_dataset(training_args.train_dataset_name,split=training_args.train_dataset_split)

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""
        # use the model chat template
        chat_template ='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nQUESTION<|im_end|>\n<|im_start|>assistant\n'
        def tokenize(element):
            
            question_sidx = element['chosen'].find('Human: ') + len('Human: ')
            question_eidx =  element['chosen'].find('Assistant: ', question_sidx)
            question = element['chosen'][question_sidx:question_eidx]
            
            temp_question = chat_template.replace('QUESTION',question)
            
            return {"input_ids": tokenizer(temp_question,padding=False,)['input_ids']}

        return dataset.map(
            tokenize,
            # batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    train_dataset = prepare_dataset(train_dataset, tokenizer)

# training ---------------------------------------------------------------------

    trainer = GRPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        train_dataset=train_dataset,
        reward_func = reward_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )


    trainer.train()
