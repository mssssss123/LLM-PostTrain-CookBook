# 17. OPD、偏好与 Agentic RL：现代后训练的复杂阶段

SFT 和 GRPO/RLVR 跑通后，才真正进入现代 post-training 的深水区：偏好优化、OPD/MOPD、Agentic RL。这些技术解决的是不同问题，不应该混成一个“更高级训练”概念。

| 技术 | 解决什么 |
|---|---|
| 偏好优化 | 开放式回答里“哪个更好” |
| OPD / MOPD | 把强教师或多个专家能力迁移到学生 |
| Agentic RL | 让模型在工具、代码、搜索、终端等环境里行动 |

本章不会假装这些都能一条命令解决。目标是让你看懂 verl 里的标准配置，并能搭出第一个可运行版本。你应该先把每个技术的训练信号分清：偏好优化看 chosen/rejected，OPD 看教师分布，Agentic RL 看环境轨迹和最终 reward。

## 1. 偏好优化：从 chosen/rejected 开始

偏好数据长这样：

```json
{
  "prompt": "解释 LoRA，面向初学者。",
  "chosen": "LoRA 可以理解为给模型加一个很小的可训练外挂...",
  "rejected": "LoRA is a low-rank adaptation method..."
}
```

在 verl 本地示例里，`full_hh_rlhf.py` 可以生成三类数据：

```bash
cd verl-main
python examples/data_preprocess/full_hh_rlhf.py \
  --split sft \
  --local_save_dir ~/data/full_hh_rlhf

python examples/data_preprocess/full_hh_rlhf.py \
  --split rm \
  --local_save_dir ~/data/full_hh_rlhf

python examples/data_preprocess/full_hh_rlhf.py \
  --split rl \
  --local_save_dir ~/data/full_hh_rlhf
```

字段区别：

| split | 字段 | 用途 |
|---|---|---|
| `sft` | `prompt`, `response` | 把偏好数据转成监督数据 |
| `rm` | `prompt`, `chosen`, `rejected` | 训练 reward model 或偏好模型 |
| `rl` | `prompt`, `reward_model` | 用 reward model 做 RL |

如果你是初学者，建议先把偏好优化放在 SFT + GRPO 之后。原因是：偏好数据提升的是开放式质量，不会替你修好数学 reward、工具环境和 token 对齐。先把“会做”训出来，再用偏好数据修“做得是否更好”。

## 2. GDPO：多维 reward 的偏好式 RL

verl 有 GDPO 示例，适合“一个回答有多个 reward 维度”的场景。脚本在：

```text
verl-main/examples/gdpo_trainer/run_qwen3_8b_fsdp.sh
```

它的关键配置：

```bash
algorithm.adv_estimator=gdpo
+algorithm.gdpo_reward_keys='["accuracy_reward", "format_reward"]'
reward.custom_reward_function.path="$REPO_ROOT/verl/utils/reward_score/rlla.py"
reward.custom_reward_function.name=compute_score
reward.reward_manager.name=gdpo
```

用 Qwen3-4B-Base 运行时：

```bash
cd verl-main
MODEL_PATH=Qwen/Qwen3-4B-Base \
DATA_DIR=$HOME/data/rlla_4k \
PROJECT_NAME=llm-posttrain-cookbook \
EXPERIMENT_NAME=qwen3-4b-base-gdpo \
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=4 \
ACTOR_LR=1e-6 \
KL_LOSS_COEF=0.001 \
bash examples/gdpo_trainer/run_qwen3_8b_fsdp.sh \
  trainer.logger='["console"]'
```

这条命令需要你先准备符合 `rlla.py` reward 预期的数据。教学阶段可以先理解它的结构，不必把 GDPO 作为第一条训练路线。多维 reward 会增加调试难度，因为你需要判断每个 reward 维度是否都在表达正确目标。

## 3. OPD：让教师在学生轨迹上教学

OPD 和普通蒸馏的差别：

- 普通蒸馏：教师先生成固定答案，学生模仿。
- OPD：学生先按当前策略生成，教师在学生真实走到的状态上给 token-level 信号。

这更适合推理、agent、多轮任务，因为学生偏离教师路径后，教师仍然能在偏离后的上下文上提供指导。这也是 OPD 比普通 off-policy 蒸馏更适合纠正学生真实错误的原因。

## 4. verl 的 OPD 配置

OPD 仍然通过 `main_ppo` 入口训练，但打开 distillation：

```bash
distillation.enabled=True
distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
distillation.nnodes=${NNODES}
distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
distillation.distillation_loss.topk=${DISTILLATION_TOPK}
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.use_policy_gradient=True
```

本地脚本：

```text
verl-main/examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
```

## 5. 单教师 OPD：Qwen3-4B-Base 学 Qwen3-8B

先准备 GSM8K 和 MATH 数据。GSM8K：

```bash
cd verl-main
python examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k
```

MATH 数据可以用：

```bash
python examples/data_preprocess/math_dataset.py \
  --local_save_dir ~/data/math
```

运行 OPD：

```bash
cd verl-main
STUDENT_MODEL=Qwen/Qwen3-4B-Base \
TEACHER_MODEL=Qwen/Qwen3-8B \
PROJECT_NAME=llm-posttrain-cookbook \
EXPERIMENT_NAME=qwen3-4b-base-opd-from-qwen3-8b \
NGPUS_PER_NODE=8 \
TEACHER_WORLD_SIZE=2 \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=64 \
ROLLOUT_TP=1 \
TEACHER_TP=1 \
ROLLOUT_GPU_MEM_UTIL=0.4 \
TEACHER_GPU_MEM_UTIL=0.4 \
ACTOR_LR=1e-6 \
DISTILLATION_LOSS_MODE=k1 \
USE_POLICY_GRADIENT=True \
DISTILLATION_TOPK=64 \
TOTAL_EPOCHS=1 \
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh \
  trainer.logger='["console"]'
```

如果你有更强教师，可以把 `TEACHER_MODEL` 换成同 Qwen3 tokenizer/vocab 的更大模型。教师必须和学生 tokenizer/vocab 兼容，否则 token-level distillation 会出问题。OPD 的信号落在 token 上，token 对不齐时，教师概率就不再对应学生的同一个动作。

一条样本在 verl OPD 里的流转：

```text
parquet.prompt
-> student rollout 生成 response
-> teacher server 在 prompt + student response 上返回 sampled-token logprob 或 top-k logprob
-> student forward 得到同位置 logprob
-> distillation_loss 计算 teacher-student KL / KL estimator
-> 可选叠加 GRPO task reward
-> actor update
```

对应配置：

| 教学代码变量 | verl 配置或字段 |
|---|---|
| `student` | `actor_rollout_ref.model.path=$STUDENT_MODEL` |
| `teacher` | `distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL` |
| `teacher_topk_ids` | teacher server 返回的 top-k token ids |
| `teacher_topk_logps` | teacher server 返回的 top-k logprobs |
| `response_mask` | 学生生成 token 的 mask |
| `loss_mode=k1` | sampled-token KL estimator |
| `loss_mode=forward_kl_topk` | top-k forward KL |
| `use_task_rewards` | 是否把环境 reward 和 distillation loss 混合 |

## 6. OPD 参数怎么选

| 参数 | 起点 | 解释 |
|---|---:|---|
| `DISTILLATION_LOSS_MODE` | `k1` | PG OPD 的常见单样本 KL 估计 |
| `USE_POLICY_GRADIENT` | `True` | 把 distill signal 当 policy-gradient 信号 |
| `DISTILLATION_TOPK` | 64 | top-k 教师分布大小，部分 loss mode 会用 |
| `TEACHER_WORLD_SIZE` | 2 到 8 | 教师推理资源池大小 |
| `TEACHER_TP` | 1 到 4 | 教师 tensor parallel |
| `ACTOR_LR` | `1e-6` | OPD 更新不要太激进 |

OPD 的失败模式：

- 教师太弱：学生学到的只是教师偏差。
- 教师太强但学生太弱：学生轨迹质量太差，教师信号利用率低。
- tokenizer/template 不一致：token-level 信号错位。
- 教师资源不足：训练大部分时间在等教师推理。
- 只看“像不像教师”，不看独立任务正确率。

## 7. MOPD：多教师能力合并

MOPD 用 `distillation.teacher_key` 做路由。常用 `data_source`：

```bash
distillation.teacher_key=data_source
+distillation.teacher_models.math.key="openai/gsm8k"
+distillation.teacher_models.math.model_path=Qwen/Qwen3-8B
+distillation.teacher_models.code.key="code_dataset"
+distillation.teacher_models.code.model_path=/models/qwen3-code-expert
+distillation.teacher_models.agent.key="agent_dataset"
+distillation.teacher_models.agent.model_path=/models/qwen3-agent-expert
```

关键规则：

- 每条样本的 `data_source` 要能匹配某个 teacher key。
- 每个 teacher 最好是同模型家族，避免 tokenizer/vocab 冲突。
- 每个领域都要有独立评估，不要只看总分。
- 多教师不是越多越好，冲突教师会让学生目标不稳定。

教学版可以先做两个教师：

| data_source | teacher |
|---|---|
| `openai/gsm8k` | 数学教师 |
| `local/tool-agent` | 工具教师 |

确认路由正确后，再扩展到 code、agent、chat 等更多教师。

## 8. Agentic RL：verl 的最小工具训练

Agentic RL 的数据需要 `agent_name`：

```bash
cd verl-main
python examples/data_preprocess/gsm8k_tool_agent_loop.py \
  --local_save_dir ~/data/gsm8k_tool
```

关键配置：

```bash
data.train_files=$HOME/data/gsm8k_tool/train.parquet
data.val_files=$HOME/data/gsm8k_tool/test.parquet
data.return_raw_chat=True
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.name=sglang
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

为什么要 `return_raw_chat=True`？因为多轮工具训练必须保留原始消息结构，让 agent loop 能在 message 级别插入 tool call 和 observation。如果只保留渲染后的 token 字符串，环境很难可靠地追加工具返回。

为什么要 `mode=async`？因为工具调用、环境执行、sandbox 返回速度不一致。同步 rollout 会被最慢轨迹拖住，GPU 容易空转。

## 9. Function tool 示例

verl 支持用 `@function_tool` 注册无状态工具。新建一个工具文件，例如 `/abs/path/my_tools.py`：

```python
from verl.tools.function_tool import function_tool


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Args:
        expression: A Python-style arithmetic expression, e.g. "(3+4)*5".
    """
    return str(eval(expression, {"__builtins__": {}}, {}))
```

训练时配置：

```bash
actor_rollout_ref.rollout.multi_turn.function_tool_path=/abs/path/my_tools.py
```

真实生产不要直接 `eval` 用户输入。这里是教学示例，生产工具必须 sandbox、timeout、限制依赖和限制副作用。工具越强，越需要明确权限边界和完整 transcript。

如果要把命令行工具接进 verl，可以先做一个严格白名单版：

```python
# /abs/path/terminal_tools.py
import shlex
import subprocess
from pathlib import Path

from verl.tools.function_tool import function_tool


WORKDIR = Path("/sandbox/task_repo").resolve()
ALLOWED = {"ls", "cat", "python", "pytest", "sed", "grep"}


@function_tool
def bash(command: str) -> dict:
    """Run a safe shell command in the task repository.

    Args:
        command: A shell-like command, e.g. "pytest -q" or "cat README.md".
    """
    argv = shlex.split(command)
    if not argv:
        return {"ok": False, "error": "empty command"}
    if argv[0] not in ALLOWED:
        return {"ok": False, "error": f"command not allowed: {argv[0]}"}

    result = subprocess.run(
        argv,
        cwd=WORKDIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-3000:],
        "stderr": result.stderr[-3000:],
    }
```

然后训练命令加：

```bash
actor_rollout_ref.rollout.multi_turn.function_tool_path=/abs/path/terminal_tools.py
```

这时模型生成的 tool call 会被 agent loop 解析，调用 `bash(command=...)`，工具返回会作为 observation 进入下一轮上下文。

## 10. Agentic RL 训练命令骨架

本地文档里有些多轮脚本路径可能随 verl 版本变化，所以这里给更稳定的 Hydra override 骨架。你可以基于 `examples/grpo_trainer/run_qwen3_4b_fsdp.sh` 追加多轮配置：

```bash
cd verl-main
MODEL_PATH=Qwen/Qwen3-4B-Base \
TRAIN_FILE=$HOME/data/gsm8k_tool/train.parquet \
TEST_FILE=$HOME/data/gsm8k_tool/test.parquet \
PROJECT_NAME=llm-posttrain-cookbook \
EXPERIMENT_NAME=qwen3-4b-base-tool-agent-grpo \
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=4 \
ROLLOUT_TP=1 \
ACTOR_LR=1e-6 \
KL_LOSS_COEF=0.001 \
bash examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  trainer.logger='["console"]'
```

如果使用 function tool：

```bash
actor_rollout_ref.rollout.multi_turn.function_tool_path=/abs/path/my_tools.py
```

如果使用 YAML tool config：

```bash
actor_rollout_ref.rollout.tool_kwargs.tools_config_file=/abs/path/tools.yaml
```

命令和代码的对应关系：

| 训练配置 | 对应代码/数据 |
|---|---|
| `data.return_raw_chat=True` | 保留 `prompt` 的 message 结构，agent loop 才能追加 tool observation |
| `actor_rollout_ref.rollout.mode=async` | 工具调用可能很慢，用异步 rollout 避免 GPU 等最慢环境 |
| `multi_turn.enable=True` | rollout 不再是一次 assistant answer，而是 action/observation 循环 |
| `agent.default_agent_loop=tool_agent` | 根据 `agent_name=tool_agent` 走工具 agent loop |
| `function_tool_path=terminal_tools.py` | 注册 `@function_tool` 里的工具函数 |
| `extra_info.tools_kwargs` | 给每条样本注入工作目录、标准答案、工具参数 |

一个工具轨迹最终会变成：

```text
user task
-> assistant tool call: bash({"command": "pytest -q"})
-> tool observation: stdout/stderr
-> assistant tool call: bash({"command": "cat failing_file.py"})
-> tool observation: file content
-> assistant final answer
-> reward function / verifier 给最终分数
```

## 11. Agentic RL 日志必须看 transcript

不要只看 reward。多轮训练必须抽 transcript：

- 模型是否产生了合法 tool call；
- tool 参数是否正确；
- observation 是否被读懂；
- 失败后是否恢复；
- 是否无限调用工具；
- 是否为了拿 reward 钻工具漏洞；
- final answer 是否忠于工具结果。

核心指标：

| 指标 | 含义 |
|---|---|
| tool call parse failure | 工具调用格式失败率 |
| turns per episode | 平均轮数 |
| tool success rate | 工具执行成功率 |
| final accuracy | 最终任务成功率 |
| response tokens | 轨迹 token 成本 |
| timeout rate | 环境超时比例 |

## 12. Qwen3 多轮 tokenization 注意点

verl 的多轮文档特别提醒：Qwen3 系列的 chat template 可能会移除最后 user 之前的 reasoning 内容。这会影响 delta-based tokenization 的一致性检查。

遇到 tokenization warning 时，不要直接关闭检查。先确认：

- 工具返回是否按 chat template 支持的 role 写入；
- thinking 内容是否被模板剥离；
- rollout token 和训练 token 是否一致；
- `multi_turn.format` 是否适配当前模型。

必要时可以临时设置：

```bash
actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
```

只有你确认差异只是空白字符时，才这样做。

## 13. 推荐学习顺序

对小白来说，复杂阶段不要并行学。顺序如下：

1. 用 `full_hh_rlhf.py --split rm` 看懂偏好数据。
2. 跑通单教师 OPD，但先用小 batch。
3. 跑通 `gsm8k_tool_agent_loop.py` 的数据构造。
4. 用最小 function tool 做 agentic RL。
5. 最后再考虑 MOPD、多工具、代码 sandbox、搜索环境。

## 14. 什么时候算“会了”

你能独立完成这些事，才算真的掌握：

- 给一个新任务写 parquet preprocess；
- 给一个新任务写 reward function；
- 判断用 SFT、GRPO、OPD 还是 Agentic RL；
- 看懂 `main_ppo` 脚本里的 `DATA/MODEL/ACTOR/ROLLOUT/REF/TRAINER`；
- 用小 run 定位 reward 全 0、KL 爆、长度失控、parse failure；
- 把 checkpoint merge 成 Hugging Face 模型；
- 用独立评估证明不是只过拟合训练 reward。

<div class="checkpoint">

**本章结论**

偏好优化解决开放质量，OPD/MOPD 解决教师能力迁移和多专家合并，Agentic RL 解决环境行动。它们都建立在前面两件事之上：数据格式正确、reward/trace 可验证。先跑通 SFT 和 GRPO，再进入这些复杂阶段，学习曲线会稳得多。

</div>
