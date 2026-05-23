# 16. GRPO/RLVR 实战：用 verl 训练可验证推理

SFT 让模型学会格式，GRPO/RLVR 让模型根据结果反馈优化策略。本章以 GSM8K 为例，训练 `Qwen/Qwen3-4B-Base` 或上一章得到的 SFT checkpoint。你可以把这一章看成第一次让模型“自己试、环境判、再更新”。

你会学到：

- RLVR parquet 长什么样；
- verl 的 GRPO 脚本如何组织配置；
- `ROLLOUT_N`、`TRAIN_BATCH_SIZE`、KL、micro batch 怎么设；
- 如何看 reward、KL、长度和 group 信号；
- 如何导出 Hugging Face checkpoint。

## 本章使用的 verl 文件

```text
verl-main/examples/data_preprocess/gsm8k.py
verl-main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh
verl-main/verl/trainer/main_ppo.py
verl-main/verl/utils/reward_score/gsm8k.py
verl-main/docs/algo/grpo.md
```

`run_qwen3_4b_fsdp.sh` 默认 `MODEL_PATH=Qwen/Qwen3-4B`。我们会覆盖成 `Qwen/Qwen3-4B-Base` 或 SFT checkpoint。

## Step 1：准备 RLVR 数据

```bash
cd verl-main
python examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k
```

抽样检查：

```bash
python - <<'PY'
import os
import pandas as pd

df = pd.read_parquet(os.path.expanduser("~/data/gsm8k/train.parquet"))
row = df.iloc[0].to_dict()
print(df.columns.tolist())
print(row["prompt"])
print(row["reward_model"])
print(row["extra_info"])
PY
```

你应该看到：

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [{"role": "user", "content": "... output the final answer after \"####\"."}],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "72"},
  "extra_info": {"split": "train", "index": 0}
}
```

这里没有 assistant 标准答案参与 loss。模型要自己生成，reward function 再判断结果。这就是 RLVR 和 SFT 的核心区别：数据里保存的是判分依据，而不是要模仿的答案。

## Step 2：理解 GRPO 脚本

`examples/grpo_trainer/run_qwen3_4b_fsdp.sh` 把配置拆成几组。

### DATA

```bash
algorithm.adv_estimator=grpo
data.train_files=${TRAIN_FILE}
data.val_files=${TEST_FILE}
data.train_batch_size=${TRAIN_BATCH_SIZE}
data.max_prompt_length=${MAX_PROMPT_LENGTH}
data.max_response_length=${MAX_RESPONSE_LENGTH}
algorithm.use_kl_in_reward=False
```

含义：

- `adv_estimator=grpo`：使用组内相对 advantage。
- `train_batch_size`：每个 global step 取多少个 prompt。
- `max_response_length`：单条回答最长生成多少 token。
- `use_kl_in_reward=False`：不把 KL 放进 reward，而是在 actor loss 中加 KL。

### MODEL

```bash
actor_rollout_ref.model.path=${MODEL_PATH}
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

`MODEL_PATH` 可以是：

- `Qwen/Qwen3-4B-Base`
- SFT 后的 checkpoint
- 已导出的 Hugging Face 模型目录

### ACTOR

```bash
actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

虽然参数名里有 `ppo`，GRPO 也使用这些 actor update 配置。`ppo_micro_batch_size_per_gpu` 主要是显存控制；`ppo_mini_batch_size` 会影响一次 rollout 数据如何切成小批更新。

### ROLLOUT

```bash
actor_rollout_ref.rollout.name=${INFER_BACKEND}
actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
actor_rollout_ref.rollout.n=${ROLLOUT_N}
```

`ROLLOUT_N` 是 GRPO 的核心参数：同一个 prompt 采样多少个答案。必须大于 1，才有组内比较。`ROLLOUT_N` 太小，advantage 噪声大；太大，采样成本会上升。

### TRAINER

```bash
trainer.project_name=${PROJECT_NAME}
trainer.experiment_name=${EXPERIMENT_NAME}
trainer.n_gpus_per_node=${NGPUS_PER_NODE}
trainer.save_freq=${SAVE_FREQ}
trainer.test_freq=${TEST_FREQ}
trainer.total_epochs=${TOTAL_EPOCHS}
```

训练输出默认保存在：

```text
checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
```

## Step 3：教学版短 run

先跑一个短实验，确认 reward 和 rollout 正常。

```bash
cd verl-main
MODEL_PATH=Qwen/Qwen3-4B-Base \
TRAIN_FILE=$HOME/data/gsm8k/train.parquet \
TEST_FILE=$HOME/data/gsm8k/test.parquet \
PROJECT_NAME=llm-posttrain-cookbook \
EXPERIMENT_NAME=qwen3-4b-base-grpo-smoke \
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=32 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=1024 \
ROLLOUT_TP=1 \
ROLLOUT_N=4 \
ROLLOUT_GPU_MEM_UTIL=0.6 \
ACTOR_LR=1e-6 \
KL_LOSS_COEF=0.001 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=5 \
TEST_FREQ=1 \
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  trainer.logger='["console"]'
```

这个配置不追求效果，只追求快。看到训练日志、reward、response length、checkpoint 正常即可。短 run 结束后要抽几条 rollout，看答案格式、reward 解析和错误样本是否符合预期。

## Step 4：更稳的 8 GPU 起点

```bash
cd verl-main
MODEL_PATH=Qwen/Qwen3-4B-Base \
TRAIN_FILE=$HOME/data/gsm8k/train.parquet \
TEST_FILE=$HOME/data/gsm8k/test.parquet \
PROJECT_NAME=llm-posttrain-cookbook \
EXPERIMENT_NAME=qwen3-4b-base-gsm8k-grpo \
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=128 \
PPO_MINI_BATCH_SIZE=64 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=1024 \
ROLLOUT_TP=1 \
ROLLOUT_N=4 \
ROLLOUT_GPU_MEM_UTIL=0.6 \
ACTOR_LR=1e-6 \
KL_LOSS_COEF=0.001 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=20 \
TEST_FREQ=5 \
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  trainer.logger='["console","wandb"]'
```

如果你从 SFT checkpoint 接着训，把 `MODEL_PATH` 指向 SFT 导出的 HF 目录：

```bash
MODEL_PATH=/path/to/qwen3-4b-base-sft-hf
```

这通常比直接从 base 跑 GRPO 更稳，因为模型已经学会了 `####` 格式。格式稳定会显著降低 parse failure，让 reward 真正反映答案对错。

## Step 5：正式实验可调参数

| 参数 | 教学起点 | 正式起点 | 作用 |
|---|---:|---:|---|
| `TRAIN_BATCH_SIZE` | 64 | 128 到 512 | 每步 prompt 数 |
| `ROLLOUT_N` | 4 | 4 到 8 | 每题采样数 |
| `PPO_MINI_BATCH_SIZE` | 32 | 64 到 256 | actor update mini-batch |
| `ACTOR_LR` | `1e-6` | `5e-7` 到 `2e-6` | RL 更新幅度 |
| `KL_LOSS_COEF` | `0.001` | `0.0005` 到 `0.005` | 控制偏离 reference |
| `MAX_RESPONSE_LENGTH` | 1024 | 1024 到 4096 | 推理输出长度 |

优先级：

1. reward 是否正确；
2. `ROLLOUT_N` 是否有组内差异；
3. KL 是否稳定；
4. response length 是否失控；
5. 再调 LR 和 batch。

## GRPO 的训练信号

同一道题采样 4 个答案：

```text
reward = [0, 1, 0, 1]
mean = 0.5
advantage = [-0.5, +0.5, -0.5, +0.5]
```

如果一组全错：

```text
reward = [0, 0, 0, 0]
advantage = [0, 0, 0, 0]
```

这一步几乎没有学习信号。初学者经常看到 reward 很低，以为“多跑就好”。其实如果模型初始化太差、格式不对或 reward parser 不匹配，继续训练也学不动。先修 SFT 初始化和 reward parser，再扩大训练。

## 一条样本在 verl GRPO 中的流转

```text
parquet.prompt + reward_model.ground_truth
-> RLHFDataset 渲染 prompt
-> rollout engine 对同一 prompt 采样 ROLLOUT_N 个 response
-> reward manager 调用 GSM8K reward
-> compute_grpo_outcome_advantage 按 prompt id 做组内归一
-> ppo_loss / policy loss 更新 actor
-> KL loss 约束 actor 不要偏离 reference
```

对应关系：

| 教学代码变量 | verl 配置或字段 |
|---|---|
| `prompt_ids` | 同一 prompt 的 index / uid |
| `rewards` | reward manager 输出的 token/sequence reward |
| `response_mask` | 只覆盖模型生成的 response token |
| `old_log_probs` | rollout 时旧策略的 logprob |
| `new_log_probs` | actor 更新时当前策略的 logprob |
| `ref_log_probs` | reference model 的 logprob |
| `ROLLOUT_N` | `actor_rollout_ref.rollout.n` |
| `KL_LOSS_COEF` | `actor_rollout_ref.actor.kl_loss_coef` |

最重要的 sanity check 是：同一 prompt 的多个 response 必须真的被分到同一组。否则 GRPO 的“组内比较”会变成随机比较，训练信号会被污染。

## 日志怎么看

重点看：

| 指标 | 解释 |
|---|---|
| `critic/score/mean` 或 reward mean | 平均正确性，应该逐渐上升 |
| `actor/ppo_kl` 或 KL 相关指标 | 策略偏离参考模型的程度 |
| `response_length/mean` | 是否越来越啰嗦 |
| `actor/pg_loss` | policy gradient loss |
| `actor/grad_norm` | 梯度是否异常爆炸 |
| `val/test_score/openai/gsm8k` | 验证集 GSM8K 分数 |

健康曲线通常是：

- reward 缓慢上升；
- KL 小幅波动；
- response length 不无意义增长；
- 验证集提升滞后于训练 reward，但方向一致。

危险信号：

- reward 突然暴涨但抽样全是格式 hack；
- KL 快速升高；
- 平均长度持续变长；
- eval 不涨，训练 reward 涨；
- parse failure 大量出现。

## 自定义 reward

如果换成自己的任务，写一个文件：

```python
# my_reward.py
import json


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        obj = json.loads(solution_str)
    except Exception:
        return 0.0

    if obj.get("answer") == ground_truth:
        return 1.0
    return 0.0
```

训练时加：

```bash
reward.custom_reward_function.path=/abs/path/my_reward.py \
reward.custom_reward_function.name=compute_score
```

注意：reward function 的输入 `solution_str` 是模型输出文本，不是 token。不要在 reward 里依赖训练器内部状态。

## 导出 checkpoint

训练中 checkpoint 通常在：

```text
checkpoints/llm-posttrain-cookbook/qwen3-4b-base-gsm8k-grpo/global_step_*/actor
```

导出为 Hugging Face 格式：

```bash
cd verl-main
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/llm-posttrain-cookbook/qwen3-4b-base-gsm8k-grpo/global_step_20/actor \
  --target_dir checkpoints/llm-posttrain-cookbook/qwen3-4b-base-gsm8k-grpo/global_step_20/actor/huggingface
```

把 `global_step_20` 换成你验证集最好的 step。不要默认最后一步最好。

## 最小实验报告模板

每次 GRPO 实验结束，记录：

```text
base/model_path:
data:
reward function:
TRAIN_BATCH_SIZE:
ROLLOUT_N:
ACTOR_LR:
KL_LOSS_COEF:
MAX_RESPONSE_LENGTH:
best checkpoint:
train reward trend:
val GSM8K:
avg response length:
代表性成功样本:
代表性失败样本:
下一步修改:
```

没有这些记录，后面很难判断一次训练为什么成功或失败。

<div class="checkpoint">

**本章结论**

GRPO/RLVR 的关键是“可验证 reward + 组内采样 + KL 控制”。用 verl 训练时，先确认 parquet 和 reward 正确，再用小 `TRAIN_BATCH_SIZE`、小 `ROLLOUT_N` 跑 smoke test，最后扩大到更稳配置。不要只看训练 reward，必须同时看 KL、长度和独立验证集。

</div>
