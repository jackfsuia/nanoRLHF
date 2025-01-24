from dataclasses import dataclass, field
import json
import time
import logging
from trl.core import masked_mean, masked_whiten
import os
from typing import Dict, Optional, List, Union, TYPE_CHECKING, Any, Callable, Tuple
import torch
from torch.utils.data import Dataset, random_split
import re
import torch
import torch.nn.functional as F
from torch import nn
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import GPTQConfig, AutoModel, Trainer
from transformers.trainer_pt_utils import LabelSmoother
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
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
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    EarlyStoppingCallback,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from accelerate.utils import DistributedType
from datasets import load_dataset
import safetensors.torch
from transformers import PreTrainedModel, Qwen2ForCausalLM
import random
from vllm import LLM, SamplingParams
import shutil
import gc

random.seed(1)
torch.manual_seed(43)
INVALID_LOGPROB = 1.0


def finetuned_value_model(
    value_model,
    policy,
    ref_policy,
    reward_func,
    ppo_dataset,
    tokenizer,
    ppo_args,
    finetune_args,
):
    def small_forward(
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
            use_cache=False,
        )

    def vllm_generate(policy, tokenizer, prompts, temperature, max_tokens):
        device = policy.device
        policy.to("cpu")
        torch.cuda.empty_cache()
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            n=1,
            max_tokens=max_tokens,
            logprobs=1,
            detokenize=False,
            ignore_eos=False,
            seed=random.randint(1, 5000),
        ) # such seed avoid overfitting
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
            outputs_ALL = llm.generate(prompts, sampling_params)
        else:
            save_model_path = "/data/temp_vllm_model"
            shutil.rmtree(save_model_path, ignore_errors=True)
            policy.save_pretrained(save_model_path)
            tokenizer.save_pretrained(save_model_path)
            llm = LLM(model=save_model_path)
            outputs_ALL = llm.generate(prompts, sampling_params)
        shutil.rmtree(save_model_path)

        response = []
        for o in outputs_ALL:
            paded_token_ids = o.outputs[0].token_ids + (tokenizer.pad_token_id,) * (
                max_tokens - len(o.outputs[0].token_ids)
            )
            tokens = torch.tensor(paded_token_ids)
            response += [tokens]

        response = torch.stack(response, dim=0)

        del llm
        torch.cuda.empty_cache()
        policy.to(device)

        return response

    class ValueDataset(Dataset):
        def __init__(
            self, query_responses, context_length, padding_mask_p1, returns, size
        ):
            self.query_responses = query_responses.to("cpu")
            self.context_length = context_length
            self.padding_mask_p1 = padding_mask_p1.to("cpu")
            self.returns = returns.to("cpu")
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):

            return {
                "input_ids": self.query_responses[idx],
                "context_length": self.context_length,
                "padding_mask_p1": self.padding_mask_p1[idx],
                "labels": self.returns[idx],
            }

    class ValueFinetuner(Trainer):

        def __init__(self, pad_token_id, *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.pad_token_id = pad_token_id

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            critic_backbone = getattr(model, model.base_model_prefix)

            output = forward(critic_backbone, inputs["input_ids"], self.pad_token_id)
            vpred_temp = model.score(output.hidden_states[-1])

            vpred = vpred_temp[:, inputs["context_length"][0] - 1 : -1].squeeze(-1)
            vpred = torch.masked_fill(vpred, inputs["padding_mask_p1"], 0)
            vf_losses1 = torch.square(vpred - inputs["labels"])
            vf_loss = 0.5 * masked_mean(vf_losses1, ~inputs["padding_mask_p1"])

            return vf_loss

        def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ):
            inputs = self._prepare_inputs(inputs)
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, return_outputs=False)
                loss = loss.mean().detach()
            return (loss, None, None)

    with torch.no_grad():

        dataloader = DataLoader(
            ppo_dataset,
            batch_size=min(finetune_args.train_data_size, len(ppo_dataset)),
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )

        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

        main_device = "cuda"
        queries = data["input_ids"].to(main_device)
        quesiton_string = tokenizer.batch_decode(queries)
        quesiton_string = [q.replace("[PAD]", "") for q in quesiton_string]
        responses = vllm_generate(
            policy,
            tokenizer,
            quesiton_string,
            ppo_args.temperature,
            max_tokens=ppo_args.response_length,
        )
        responses_decoded = tokenizer.batch_decode(responses)
        scores = reward_func([i + j for i, j in zip(quesiton_string, responses_decoded)], tokenizer.eos_token).to(main_device)

        gc.collect()
        responses = responses.to(main_device)
        ref_policy.to(main_device)
        policy.to(main_device)
        policy.eval()

        context_length = queries.shape[1]
        logprobs = []
        ref_logprobs = []
        sequence_lengths = []
        query_responses = []

        local_rollout_forward_batch_size = (28 * 2316 // (context_length + ppo_args.response_length))


        # generating value dataset----------------------------------------------------------------------
        for i in tqdm(range(0, queries.shape[0], local_rollout_forward_batch_size),desc="Generating value finetune dataset"):
            query = queries[i : i + local_rollout_forward_batch_size]
            query_response = torch.cat((query, responses[i : i + local_rollout_forward_batch_size]), dim=1)

            response = query_response[:, context_length:]

            ############### 输出比值部分的 log_prob
            logprob = small_forward(policy, query_response, tokenizer.pad_token_id)
            logprob = logprob[0][:, context_length - 1 : -1].clone()
            gc.collect()
            torch.cuda.empty_cache()
            logprob /= ppo_args.temperature
            logprob = F.log_softmax(logprob, dim=-1)
            logprob = torch.gather(logprob, 2, response.unsqueeze(-1)).squeeze(-1)

            ###############输出单步价值 ref_logprob,value
            ref_logprob = small_forward(ref_policy, query_response, tokenizer.pad_token_id)
            ref_logprob = ref_logprob[0][:, context_length - 1 : -1].clone()
            gc.collect()
            torch.cuda.empty_cache()
            ref_logprob /= ppo_args.temperature
            ref_logprob = F.log_softmax(ref_logprob, dim=-1)
            ref_logprob = torch.gather(ref_logprob, 2, response.unsqueeze(-1)).squeeze(-1)

            # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
            postprocessed_response = response
            if (ppo_args.stop_token_id is not None): 
                # handle the edge case when stop_token_id exists but is 0
                postprocessed_response = truncate_response(ppo_args.stop_token_id, tokenizer.pad_token_id, response)
            # Response Processing 2. run reward model on the truncated responses
            # postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            sequence_length = (first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1)

            query_responses.append(query_response)
            logprobs.append(logprob)
            ref_logprobs.append(ref_logprob)
            sequence_lengths.append(sequence_length)

        ref_policy.to("cpu")
        policy.to("cpu")

        query_responses = torch.cat(query_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        # sequence_lengths是最后一个生成token的下标，sequence_lengths_p1是这个位置的后一个位置
        del ref_logprob
        gc.collect()
        torch.cuda.empty_cache()
        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.

        # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(
            responses.shape[1], device=responses.device
        ).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_p1 = sequence_lengths + 1
        padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))

        # 4. compute rewards，reward的长度是value的长度，稀疏reward算在EOS状态上，即reward(s_EOS,a) = 稀疏reward

        kl = logprobs - ref_logprobs
        non_score_reward = -ppo_args.kl_coef * kl
        rewards = non_score_reward.clone()
        actual_start = torch.arange(rewards.size(0), device=rewards.device)
        # 原因是value函数的拟合和reward在哪一步产生无关，只要在value后面的状态产生就好
        actual_end = torch.where(
            sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths
        )

        # 4.5 shape reward
        normal_scores = scores
        rewards[[actual_start, actual_end]] += normal_scores

        # 5. whiten rewards
        if ppo_args.whiten_rewards:
            rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=True)
            rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

        # 6. compute advantages and returns
        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
            lastgaelam = rewards[:, t] + ppo_args.gamma * lastgaelam
            advantages_reversed.append(lastgaelam)
        returns = torch.stack(advantages_reversed[::-1], axis=1)

    finetune_dataset = ValueDataset(query_responses, context_length, padding_mask_p1, returns, queries.shape[0])

    train_size = int(len(finetune_dataset) * finetune_args.train_split_rate)
    test_size = len(finetune_dataset) - train_size
    train_dataset, eval_dataset = random_split(finetune_dataset, [train_size, test_size])

    value_model.to("cuda")

    trainer = ValueFinetuner(
        data_collator=default_data_collator,
        pad_token_id=tokenizer.pad_token_id,
        model=value_model,
        tokenizer=tokenizer,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    value_model.to("cpu")

    shutil.rmtree(finetune_args.output_dir, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    return value_model