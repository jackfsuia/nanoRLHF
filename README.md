# nanoRLHF
RLHF experiments on a single A100 40G GPU.
# Features
Compared to [trl](https://github.com/huggingface/trl), nanoRLHF

1. is **much more efficient** running on a one single GPU. This done by using vllm to generate samples, models' alternate offloading, and some minor modifications of trl codes.
2. provided **GRPO and Remax implementations**, and a slightly different RLOO where we abandon some examples to save time.
3. **seletively enable advantage whiten** according to the algorithm you choose. To my understanding, advantage whiten is used to provide a simple dynamic baseline for advantage function. For algorithms like GRPO, PPO, RLOO, Remax, which themself have provided a baseline, we disable the advantage whiten by default. For Reinforce, we enable it.
4. use a **changing random seed in vllm generation** to avoid overfitting.
5. provide value model initialization for PPO, this is extremely crutial for successful training of PPO, especially for rule-based reward, where you have no reward model to initialize your value model. That might cause the training to be compeletely stagnant.
# Usage
Take GRPO for example, 
```bash
cd nanoRLHF/grpo
```
```
python grpo.py
```
   
# Default Setting
policy model : Qwen/Qwen2.5-1.5B-Instruct

reward model/function : OpenAssistant/reward-model-deberta-v3-large-v2

dataset: Anthropic/hh-rlhf

...

ALL setting is on the file you run.
# Performance
The training throughput is approximately 1s /episode with the default settings. Reward results are as follows (not finished the whole run):
![performance](docs/perf.png)
# Acknowledgement
The code is adapted from [trl](https://github.com/huggingface/trl), but way more **efficient**, more **flexible** reward function, specially designed for researchers that want to try small RLHF experiments quick on a single GPU.
