Compared to trl, nanoRLHF

1. is **much more efficient** running on a one single GPU. This done by using vllm to generate samples, models' alternate offloading, and some other modifications of trl codes.
2. provided **GRPO and Remax implementations**, and a slightly different RLOO where we abandon some examples to save time.
3. **seletively enable advantage whiten according** to the algorithm you choose. To my understanding, advantage whiten is used to provide a simple dynamic baseline for advantage function. For algorithms like GRPO, PPO, RLOO, Remax, which themself have provided a baseline, we disable the advantage whiten by default. For reinforce, we enable it.
4. use a **changing random seed in vllm generation** to avoid overfitting.
