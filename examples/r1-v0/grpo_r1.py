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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    EarlyStoppingCallback,
)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import ModelConfig, PPOConfig, PPOTrainer
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
import multiprocessing
from datasets import Dataset
import json, os
from typing import Dict, List, Optional, Tuple, Union
from utils.toolkit_for_MATH.latex_answer_check import latex_answer_check as latex_equiv
from utils.eval.eval_script import is_correct as is_correct_dk, eval_ocwcourses
from tqdm import tqdm
from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional
from accelerate.utils import broadcast, gather_object
from grpo_r1_trainer import GRPOTrainer
from memory_profiler import profile
import psutil
from multiprocessing import Process, set_start_method, Queue, get_start_method
import sys
import random

random.seed(0)
torch.manual_seed(42)

set_start_method("spawn", force=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

INVALID_LOGPROB = 1.0


# ALL parameters -----------------------------------------------------------------------------------------------------------

base_model_name_or_path = "Qwen/Qwen2-1.5B"
experiment_name = "r1-v0"
os.environ["WANDB_PROJECT"] = experiment_name

@dataclass
class GRPOConfig(PPOConfig):
    memory_log: str = field(
        default=f"{base_model_name_or_path}/{experiment_name}/memory_whiten.log"
    )
    save_value_model: bool = field(default=True)
    early_stopping_patience: int = field(default=1000000)
    accuracy_before_train: bool = field(default=True)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] = field(
        default_factory=lambda: ["embed_tokens", "lm_head", "score"]
    )

    lora_bias: str = field(default="none")
    q_lora: bool = field(default=False)

    train_dataset_name: str = field(default="meta-math/MetaMathQA")
    train_dataset_split: str = field(default="train[:100%]")
    accuracy_dataset_name: str = field(default="HuggingFaceH4/MATH-500")

    advantage_whiten: bool = field(default=False)

    grpo_sample_N: int = field(default=4)


training_args = GRPOConfig(
    exp_name="r1-v0",
    whiten_rewards=False,
    kl_coef=0.0, #0
    cliprange=0.2,  
    temperature=0.75,#0.75
    learning_rate=9e-6,  # 9e-6
    warmup_steps=0,  # 4
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    response_length=8000,#1500
    ## mini_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    ## batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_mini_batches
    per_device_train_batch_size=4,  # 4
    gradient_accumulation_steps=8,  # 8
    num_mini_batches=16,  # 16
    num_ppo_epochs=1,
    total_episodes=250000,  # 100000
    local_rollout_forward_batch_size=16,
    bf16=True,
    gradient_checkpointing=True,
    missing_eos_penalty=None,
    report_to="wandb",
    sft_model_path=base_model_name_or_path,
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=6,
    log_level="info",
    output_dir=f"{base_model_name_or_path}/{experiment_name}",
    logging_dir=f"{base_model_name_or_path}/{experiment_name}/logs",
    logging_strategy="steps",
    logging_steps=1,
    metric_for_best_model="eval_objective/rlhf_reward_old",
    greater_is_better=True,
    load_best_model_at_end=True,
    stop_token="eos",
    # torch_compile=True,
)


# define correctness function ------------------------------------

def call_with_timeout(func, *args, timeout=0.015, **kwargs):
 # from .utils/eval/, this is to prevent long computing, such as 2^(2^100000). 0.015s is the magic time for you to decide.
    output_queue = multiprocessing.Queue()  
    process_args = args + (output_queue,)  
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)  
    process.start()  
    process.join(timeout)  
  
    if process.is_alive():  
        process.terminate()
        process.join()  
        return False  
  
    return output_queue.get()

def get_boxed(box_answer):
    pos = box_answer.find("boxed{")
    if pos == -1:
        return ""
    pos = pos + len("boxed{")
    box_answer = box_answer[pos:]
    lef = 1
    right = 0
    right_pos = 0
    for i in range(len(box_answer)):
        if box_answer[i] == "}":
            right += 1
        if box_answer[i] == "{":
            lef += 1
        if lef == right:
            right_pos = i
            break
    box_answer = box_answer[:right_pos]
    box_answer = box_answer.replace(" ", "")
    return box_answer


def iscorrect(answer, correct_answer):
    if answer == "":
        return False
    if answer.strip() == correct_answer.strip():
        return True
    if call_with_timeout(latex_equiv, answer, correct_answer) or call_with_timeout(is_correct_dk, {"prediction": answer, "answer": correct_answer}):
        return True

    return False


# prepare dataset hash map for training accuracy evaluation and MATH-500 accuracy evaluation ------------------------------------
template = "# Question:\nQUESTION\nPlease reason step by step, and put your final answer within \\boxed{}.\n# Answer:\n" # my template
def get_MetaMathQA_answers(response):
    a_idx = response.find('The answer is: ') + len('The answer is: ')
    answer = response[a_idx:].strip()
    return answer

train_dataset = load_dataset(training_args.train_dataset_name, split=training_args.train_dataset_split)
accuracy_dataset = load_dataset(training_args.accuracy_dataset_name)
accuracy_dataset = accuracy_dataset['test']
train_dataset_index = {}

for qa in tqdm(train_dataset,desc="train dataset eval indexing"):
    train_dataset_index[qa["query"]] = get_MetaMathQA_answers(qa["response"])

accuracy_prompts = []
accuracy_solutions = []
for qa in tqdm(accuracy_dataset,desc="eval dataset eval indexing"):
    accuracy_prompts.append(template.replace("QUESTION", qa['problem']))
    accuracy_solutions.append(get_boxed(qa["solution"]))
del accuracy_dataset
# define reward function, reward =1 if correct, otherwise 0. ------------------------------------

def reward_func(pmt_and_responses, responses_ids, tokenizer):

    rewards = torch.zeros(len(pmt_and_responses))

    for p_i, pmt_and_response in tqdm(enumerate(pmt_and_responses),desc='rewarding'):

        problem_start_idx = len("# Question:\n")
        problem_end_idx = pmt_and_response.find("\nPlease reason step by step, and")
        problem = pmt_and_response[problem_start_idx:problem_end_idx]

        solu_idx = pmt_and_response.find("\n# Answer:\n", problem_end_idx) + len(
            "\n# Answer:\n"
        )
        endix = pmt_and_response.find(tokenizer.eos_token, solu_idx)
        solution = pmt_and_response[solu_idx:endix]

        if endix == -1:
            solution = pmt_and_response[solu_idx:]

        solution = get_boxed(solution)

        if iscorrect(solution, train_dataset_index[problem]):
            rewards[p_i] = 1
    return rewards

# define accuracy evaluation on MATH 500.-------------------------------------------------------------
def accuracy_func(model, args):

    num_of_question = len(accuracy_prompts)
    device = model.policy.device
    model.to("cpu")
    policy = model.policy
    torch.cuda.empty_cache()
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        n=1,
        max_tokens=args.response_length,
        ignore_eos=False,
        seed=42,
    )
    save_model_path = "/data/temp_vllm_model"
    shutil.rmtree(save_model_path, ignore_errors=True)

    if policy.peft_type:
        save_adapter_path = save_model_path + "/adapter"
        policy.save_pretrained(save_adapter_path)
        temp_policy = AutoModelForCausalLM.from_pretrained(
            policy.name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        temp_policy = PeftModel.from_pretrained(temp_policy, save_adapter_path)
        temp_policy.merge_and_unload()
        save_full_model_path = save_model_path + "/full_model"
        temp_policy.base_model.model.save_pretrained(save_full_model_path)
        del temp_policy
        tokenizer.save_pretrained(save_full_model_path)
        llm = LLM(model=save_full_model_path)
        outputs_ALL = llm.generate(accuracy_prompts, sampling_params)
    else:
        save_model_path = "/data/temp_vllm_model"
        shutil.rmtree(save_model_path, ignore_errors=True)
        policy.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
        llm = LLM(model=save_model_path)
        outputs_ALL = llm.generate(accuracy_prompts, sampling_params)
    shutil.rmtree(save_model_path)

    outputs_sample = []
    for o in outputs_ALL:
        for o2 in o.outputs:
            outputs_sample.append(o2.text)
            break

    is_corrrect_res = torch.zeros(num_of_question)

    for p_i, response in enumerate(outputs_sample):

        answer = get_boxed(response)

        if iscorrect(answer, accuracy_solutions[p_i]):
            is_corrrect_res[p_i] = 1

    accuracy = torch.mean(is_corrrect_res).item()
    del llm
    torch.cuda.empty_cache()

    model.to(device)
    print(f"acc = {accuracy}\n")
    return accuracy


if __name__ == "__main__":

    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        padding_side="left",
        trust_remote_code=False,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if training_args.use_lora:

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=training_args.modules_to_save,  # This argument serves for adding new tokens.
        )
        if training_args.q_lora:
            policy = prepare_model_for_kbit_training(
                policy, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

  
        policy = get_peft_model(policy, lora_config)

        policy.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            policy.enable_input_require_grads()

# load dataset ---------------------------------------------------------

    train_dataset = load_dataset(training_args.train_dataset_name,split=training_args.train_dataset_split) 

    def prepare_dataset(dataset, tokenizer):
        def tokenize(element):
            outputs = tokenizer(
                [
                    template.replace("QUESTION", i)
                    for i in element["query"]
                ],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
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
        reward_func=reward_func,
        accuracy_func=accuracy_func,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        ],
    )
    trainer.train()
