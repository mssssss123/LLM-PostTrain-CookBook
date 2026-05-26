# 术语、论文与速查表

这一页用来快速查概念、方法选择和进一步阅读资料。

## 方法速查

| 缩写 | 全称 | 一句话 |
|---|---|---|
| SFT | Supervised Fine-Tuning | 用示范答案训练模型 |
| RL | Reinforcement Learning | 用 reward 更新模型策略 |
| RLVR | RL with Verifiable Rewards | 用可程序验证的 reward 做 RL |
| RLHF | RL from Human Feedback | 用人类偏好训练 reward model，再 RL |
| RM | Reward Model | 给回答打分的模型 |
| PPO | Proximal Policy Optimization | 限制策略更新幅度的 RL 算法 |
| GRPO | Group Relative Policy Optimization | 同题多样本组内比较的 RL 方法 |
| DPO | Direct Preference Optimization | 直接用 chosen/rejected 偏好对优化 |
| ORPO | Odds Ratio Preference Optimization | 将监督学习和偏好项结合 |
| SimPO | Simple Preference Optimization | 简化偏好优化目标 |
| KTO | Kahneman-Tversky Optimization | 使用好/坏单样本反馈的偏好优化 |
| OPD | On-Policy Distillation | 学生当前策略生成轨迹，教师给分布信号 |
| MOPD | Multi-Teacher On-Policy Distillation | 多个领域教师共同蒸馏一个统一学生模型 |
| SDFT | Self-Distillation Fine-Tuning | 用自蒸馏约束减少遗忘 |
| Agentic RL | Agentic Reinforcement Learning | 在工具/搜索/代码/终端等环境中训练模型行动 |
| GRM | Generative Reward Model | 用生成式模型按 rubric 评价复杂输出或轨迹 |
| ORM | Outcome Reward Model | 根据最终结果给分的奖励模型 |
| PARL | Parallel-Agent Reinforcement Learning | 训练 orchestrator 动态创建和调度并行子 agent |
| KL | Kullback-Leibler Divergence | 衡量两个分布差异 |
| LoRA | Low-Rank Adaptation | 参数高效微调方法 |
| Renderer | Chat/template adapter | 消息、token、mask、解析的适配层 |

## 概念到代码速查

| 概念点 | 配套代码位置 |
|---|---|
| Base eval runner | [1. Post-Training 全景与现代流水线](./01-overview.md) |
| 数据路由和多教师路由 | [1. Post-Training 全景与现代流水线](./01-overview.md) |
| SFT labels / loss mask | [2. 数据、模板与 Renderer](./02-data-rendering.md) |
| SFT loss | [3. SFT](./03-sft.md) |
| REINFORCE / KL / PPO clipped loss | [4. RL 基础](./04-rl-foundations.md) |
| GRPO advantage / policy loss | [5. GRPO 与 RLVR](./05-grpo-rlvr.md) |
| 代码 sandbox reward | [5. GRPO 与 RLVR](./05-grpo-rlvr.md) |
| 搜索工具和搜索 reward | [5. GRPO 与 RLVR](./05-grpo-rlvr.md) |
| Reward model / DPO / SimPO / ORPO / KTO | [6. 偏好优化](./06-preference-alignment.md) |
| Off-policy distillation / OPD / SDFT | [7. 蒸馏与 OPD](./07-distillation-opd.md) |
| Code agent / terminal agent / search agent / GUI reward / multi-agent | [8. Agentic RL](./08-tools-multiturn-agent.md) |
| Inline eval / pass@k / judge / 错误分析 | [9. 评估体系](./09-evaluation.md) |
| LoRA layer | [10. LoRA 与超参](./10-hyperparams-lora.md) |
| Checkpoint record / HF smoke test / A/B log | [11. Checkpoint 与部署](./11-checkpoints-deployment.md) |
| 实验 schema / 数据校验 / anomaly detector / sweep | [12. 实验手册](./12-experiment-playbook.md) |
| 自定义 SFT/RLVR/Agentic/Preference parquet | [14. 数据、Reward 与 parquet](./14-verl-data-reward.md) |
| verl SFT/GRPO/OPD 样本流转 | [15](./15-verl-sft-qwen3.md)、[16](./16-verl-grpo-rlvr.md)、[17](./17-verl-opd-agent-preference.md) |

## 方法选择

| 目标 | 建议路线 |
|---|---|
| 教模型固定输出格式 | SFT |
| 让回答更符合人类偏好 | SFT + DPO |
| 建立传统助手模型 | SFT + DPO/RLHF |
| 提升数学推理 | SFT warm start + GRPO/RLVR |
| 提升代码能力 | SFT + sandbox RL |
| 训练搜索工具使用 | tool-call SFT + 多轮 RL |
| 训练代码/终端 agent | agent trajectory SFT + Agentic RL + sandbox eval |
| 训练并行多 agent | PARL / orchestrator RL |
| 迁移强模型能力 | off-policy distillation 或 OPD |
| 减少新任务遗忘 | SDFT + 回归评估 |
| 做多轮 Agent | verl agent loop + sandbox + RL/OPD |

## 训练检查清单

### 数据

- 数据 schema 有校验。
- 训练集和评估集去重。
- 随机样本人工抽检。
- 长度分布可视化。
- renderer decode 正常。
- mask 覆盖范围正确。

### SFT

- 小样本过拟合成功。
- train/test loss 都记录。
- 定期采样输出。
- 最终行为用独立集评估。

### RL

- reward 函数有单元测试。
- rollout transcript 可读。
- parse failure 有明确处理。
- KL、长度、reward、失败率都记录。
- eval 集独立于训练 reward。

### DPO

- chosen/rejected 差异明确。
- 检查长度偏差。
- `dpo_beta` 从 0.1 附近试。
- 监控 win-rate、长度、拒绝率。

### OPD / 蒸馏

- 教师模型先评估。
- 学生有足够初始化。
- 教师调用成本可控。
- 同时评估“像教师”和“任务正确”。

## verl 路径索引

| 主题 | 参考路径 |
|---|---|
| verl 总览 | `verl-main/README.md` |
| 数据准备说明 | `verl-main/docs/preparation/prepare_data.rst` |
| reward function 说明 | `verl-main/docs/preparation/reward_function.rst` |
| SFT 脚本 | `verl-main/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh` |
| SFT 数据生成 | `verl-main/examples/data_preprocess/gsm8k_multiturn_sft.py` |
| GRPO 脚本 | `verl-main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh` |
| GRPO 算法说明 | `verl-main/docs/algo/grpo.md` |
| GSM8K RL 数据生成 | `verl-main/examples/data_preprocess/gsm8k.py` |
| Agentic RL 说明 | `verl-main/docs/start/agentic_rl.rst` |
| 多轮工具说明 | `verl-main/docs/sglang_multiturn/multiturn.rst` |
| 工具数据生成 | `verl-main/examples/data_preprocess/gsm8k_tool_agent_loop.py` |
| OPD 脚本 | `verl-main/examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh` |
| OPD 算法说明 | `verl-main/docs/algo/opd.md` |
| GDPO 脚本 | `verl-main/examples/gdpo_trainer/run_qwen3_8b_fsdp.sh` |
| 偏好数据生成 | `verl-main/examples/data_preprocess/full_hh_rlhf.py` |
| Checkpoint 转 HF | `verl-main/docs/advance/checkpoint.rst` |

## 常用命令模板

### SFT

```bash
cd verl-main
python examples/data_preprocess/gsm8k_multiturn_sft.py \
  --local_save_dir ~/data/gsm8k_sft

MODEL_PATH=Qwen/Qwen3-4B-Base \
USE_PEFT=1 \
LORA_RANK=32 \
LR=1e-4 \
bash examples/sft/gsm8k/run_qwen3_8b_fsdp.sh \
  8 \
  ~/checkpoints/qwen3-4b-base-sft \
  data.train_files=$HOME/data/gsm8k_sft/train.parquet \
  data.val_files=$HOME/data/gsm8k_sft/test.parquet
```

### GRPO/RLVR

```bash
cd verl-main
python examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k

MODEL_PATH=Qwen/Qwen3-4B-Base \
TRAIN_FILE=$HOME/data/gsm8k/train.parquet \
TEST_FILE=$HOME/data/gsm8k/test.parquet \
TRAIN_BATCH_SIZE=128 \
PPO_MINI_BATCH_SIZE=64 \
ROLLOUT_N=4 \
ACTOR_LR=1e-6 \
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh
```

### 偏好数据

```bash
cd verl-main
python examples/data_preprocess/full_hh_rlhf.py \
  --split rm \
  --local_save_dir ~/data/full_hh_rlhf
```

### OPD

```bash
cd verl-main
STUDENT_MODEL=Qwen/Qwen3-4B-Base \
TEACHER_MODEL=Qwen/Qwen3-8B \
TEACHER_WORLD_SIZE=2 \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=64 \
DISTILLATION_LOSS_MODE=k1 \
USE_POLICY_GRADIENT=True \
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
```

### Agentic RL 数据

```bash
cd verl-main
python examples/data_preprocess/gsm8k_tool_agent_loop.py \
  --local_save_dir ~/data/gsm8k_tool
```

实际命令需要根据 GPU 数、显存、verl 版本和数据配置调整。

## 进一步阅读

### 基础与 RLHF

- InstructGPT: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### 偏好优化

- DPO: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- ORPO: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- KTO: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- SimPO: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)

### 推理与 RLVR

- DeepSeekMath / GRPO: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- Qwen3: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- Process supervision: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

### 工业后训练公开流程

- Llama 3: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- OpenAI InstructGPT: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- OpenAI o1: [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- Anthropic: [Building effective agents](https://www.anthropic.com/research/building-effective-agents)
- Anthropic: [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)

### 工具与环境

- Search-R1: [Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)
- Verifiers: [Prime Intellect Verifiers](https://github.com/primeintellect-ai/verifiers)
- SandboxFusion: [SandboxFusion](https://github.com/bytedance/SandboxFusion)
- DeepSeek-V4: [Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- MiMo-V2-Flash: [Technical Report](https://arxiv.org/pdf/2601.02780)
- Kimi K2.5: [Visual Agentic Intelligence](https://arxiv.org/pdf/2602.02276)
- GLM-5: [from Vibe Coding to Agentic Engineering](https://arxiv.org/pdf/2602.15763)

### 蒸馏与持续学习

- Thinking Machines Lab: [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- SDFT: [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)

## 最后提醒

Post-training 的方法更新很快，但底层检查项变化很慢：

- 数据是否对；
- 模板是否对；
- loss/reward 是否代表目标；
- 评估是否独立；
- 日志是否能解释行为；
- checkpoint 是否可复现。

把这些做好，再追新算法才有意义。
