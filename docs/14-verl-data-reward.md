# 14. verl 数据、Reward 与 parquet

训练代码能不能跑通，第一关不是算法，而是数据。verl 的核心约定是：训练样本一般先转成 parquet，训练器再按字段读取、套 chat template、生成 token、计算 loss 或 reward。你可以把 parquet 看成训练框架和数据工程之间的合同：字段写对了，训练器才知道该做 SFT、RLVR、Agentic RL 还是偏好训练。

这一章只回答三个问题：

- SFT、RL、Agentic RL、偏好数据分别长什么样？
- reward 在 verl 里从哪里来？
- 初学者如何检查 parquet 是否正确？

## 为什么是 parquet

parquet 适合大规模训练：

- 列式存储，读取快；
- 支持嵌套结构，能放 `prompt`、`messages`、`extra_info`；
- 和 Hugging Face `datasets`、pandas、pyarrow 都能互转；
- 适合分布式训练按行切分。

你可以把一条样本理解为 Python dict，最后保存成 parquet。初学阶段不要急着做大数据集，先手写 3 到 5 条样本并 decode 检查，比直接转几万条更可靠。

配套代码：最小 parquet 写入。

```python
import os
import pandas as pd


def save_parquet(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.expanduser(path))
```

## SFT 数据：`messages`

SFT 最常用字段是 `messages`。本地脚本 `examples/data_preprocess/gsm8k_multiturn_sft.py` 会生成：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Natalia sold clips ... Let's think step by step and output the final answer after \"####\"."
    },
    {
      "role": "assistant",
      "content": "Natalia sold 48/2 = 24 clips in May. ... #### 72"
    }
  ]
}
```

SFT 脚本中对应的关键配置是：

```bash
data.messages_key=messages
```

训练器会读取 `messages`，应用模型 tokenizer 的 chat template，再只对需要学习的 assistant token 计算交叉熵。对初学者来说，最重要的是确认：

- user 内容没有被当作目标答案训练；
- assistant 答案没有被截断；
- chat template 和部署时一致；
- `messages` 的 role 只用模型支持的角色。

准备 SFT 数据：

```bash
cd verl-main
python examples/data_preprocess/gsm8k_multiturn_sft.py \
  --local_save_dir ~/data/gsm8k_sft
```

配套代码：把自己的中文问答转成 verl SFT 数据。

```python
rows = [
    {
        "messages": [
            {"role": "system", "content": "你是一个严谨的大模型后训练助教。"},
            {"role": "user", "content": "什么是 SFT？"},
            {"role": "assistant", "content": "SFT 是用高质量示范答案训练模型，让模型学会按指令回答。"},
        ]
    }
]

save_parquet(rows, "~/data/my_sft/train.parquet")
```

这条数据可以接到 SFT 脚本：

```bash
data.train_files=$HOME/data/my_sft/train.parquet
data.messages_key=messages
```

## RL/RLVR 数据：`prompt` + `reward_model`

GRPO/RLVR 不需要标准答案文本作为监督目标，而是需要 prompt 和可判分信息。本地 `examples/data_preprocess/gsm8k.py` 生成的样本是：

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [
    {
      "role": "user",
      "content": "Natalia sold clips ... Let's think step by step and output the final answer after \"####\"."
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "72"
  },
  "extra_info": {
    "split": "train",
    "index": 0,
    "answer": "Natalia sold 48/2 = 24 clips in May. ... #### 72",
    "question": "Natalia sold clips to 48 of her friends in April..."
  }
}
```

字段含义：

| 字段 | 作用 |
|---|---|
| `data_source` | 用于选择 reward function，也常用于 OPD/MOPD 教师路由 |
| `prompt` | 模型要看到的聊天消息 |
| `ability` | 任务类别，方便日志和数据混合 |
| `reward_model.ground_truth` | reward 需要的标准答案 |
| `extra_info` | 调试、trace、工具参数、样本 id |

准备 RL 数据：

```bash
cd verl-main
python examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k
```

配套代码：把自己的可验证题目转成 RLVR 数据。

```python
rows = [
    {
        "data_source": "local/arithmetic",
        "prompt": [
            {"role": "user", "content": "计算 17 * 23，最后用 #### <答案> 输出。"}
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "391"},
        "extra_info": {"split": "train", "index": 0, "question": "17 * 23"},
    }
]

save_parquet(rows, "~/data/arithmetic/train.parquet")
```

对应 reward function 要能解析 `#### 391`，并和 `ground_truth` 比较。也就是说，RLVR 数据的重点不是提供标准回答文本，而是提供让环境判分所需的信息。

## Agentic RL 数据：`agent_name` + 工具参数

多轮工具训练还需要告诉 rollout 使用哪个 agent loop。`examples/data_preprocess/gsm8k_tool_agent_loop.py` 会额外加入 `agent_name`：

```json
{
  "data_source": "openai/gsm8k",
  "agent_name": "tool_agent",
  "prompt": [
    {
      "role": "system",
      "content": "You are a math expert. ... You should use the `calc_gsm8k_reward` tool ..."
    },
    {
      "role": "user",
      "content": "Natalia sold clips ... "
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "72"
  },
  "extra_info": {
    "need_tools_kwargs": true,
    "tools_kwargs": {
      "calc_gsm8k_reward": {
        "create_kwargs": {
          "ground_truth": "72"
        }
      }
    }
  }
}
```

这里的重点不是 GSM8K 本身，而是模式：

- `agent_name` 决定 rollout 走普通单轮还是工具 agent loop；
- `prompt` 包含 system 工具协议；
- `extra_info.tools_kwargs` 把每条样本的环境参数传给工具；
- reward 仍然可以用最终答案或工具执行结果计算。

准备工具数据：

```bash
cd verl-main
python examples/data_preprocess/gsm8k_tool_agent_loop.py \
  --local_save_dir ~/data/gsm8k_tool
```

配套代码：构造一个命令行 agent 任务数据。这里不直接训练模型，只展示 parquet 需要携带什么。

```python
rows = [
    {
        "data_source": "local/terminal",
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": "你可以调用 bash(command) 在仓库中排查问题，最后给出 final。"},
            {"role": "user", "content": "运行测试并修复失败。"},
        ],
        "ability": "terminal_agent",
        "reward_model": {"style": "rule", "ground_truth": "tests_pass"},
        "extra_info": {
            "split": "train",
            "index": 0,
            "repo_path": "/sandbox/tasks/task_0001",
            "tools_kwargs": {
                "bash": {
                    "create_kwargs": {"workdir": "/sandbox/tasks/task_0001", "timeout_s": 10}
                }
            },
        },
    }
]

save_parquet(rows, "~/data/terminal_agent/train.parquet")
```

关键点：工具环境所需的每条样本参数要放进 `extra_info`，例如仓库路径、标准答案、工具超时、隐藏测试路径。如果这些参数散落在外部脚本里，训练结果就很难复现。

## 偏好数据：`prompt/chosen/rejected`

偏好数据用于 reward model、DPO、online DPO 或通用对齐。`examples/data_preprocess/full_hh_rlhf.py --split rm` 会生成：

```json
{
  "prompt": "Human: Can you help me ...",
  "chosen": "Assistant: Sure, here is a safer and more helpful answer...",
  "rejected": "Assistant: I cannot help."
}
```

这类数据表达的是“chosen 比 rejected 更好”。偏好可以来自人工、规则、LLM judge 或已有 reward model。

准备偏好数据示例：

```bash
cd verl-main
python examples/data_preprocess/full_hh_rlhf.py \
  --split rm \
  --local_save_dir ~/data/full_hh_rlhf
```

配套代码：自己构造偏好对。

```python
rows = [
    {
        "prompt": "用初学者能懂的话解释 GRPO。",
        "chosen": "GRPO 会让模型对同一道题生成多个答案，再奖励组内表现更好的答案。",
        "rejected": "GRPO 是一种复杂优化算法，细节略。",
    }
]

save_parquet(rows, "~/data/my_preference/train.parquet")
```

偏好数据的坑比格式更多：

- chosen 不要只是更长；
- rejected 不要全是明显垃圾；
- 边界类偏好和帮助性偏好要分开评估；
- prompt 分布要接近未来真实使用场景；
- 如果用 LLM judge，要保留 judge prompt 和版本。

## Reward 在哪里计算

GRPO/RLVR 的 reward 一般由 reward function 计算。verl 支持内置 reward，也支持自定义函数。

自定义函数签名：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    ...
    return score
```

最小数学 reward：

```python
import re


def extract_final_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?[0-9.,]+)", text)
    if match is None:
        return None
    return match.group(1).replace(",", "").strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = extract_final_answer(solution_str)
    if pred is None:
        return 0.0
    return 1.0 if pred == str(ground_truth).strip() else 0.0
```

训练时指定：

```bash
reward.custom_reward_function.path=/path/to/my_reward.py
reward.custom_reward_function.name=compute_score
```

GSM8K 示例默认已经有内置 reward。只有当你换成自己的任务时，才需要写自定义 reward。

## 多 reward：准确率、格式、成本

真实任务通常不止一个 reward。一个数学任务可以拆成：

| reward | 例子 | 权重建议 |
|---|---|---:|
| accuracy | 最终答案正确 | 主要权重 |
| format | 是否包含 `####` 或 `\boxed{}` | 小权重 |
| length | 输出是否过长 | 惩罚项 |
| boundary | 是否触发边界/权限问题 | 强约束 |

不要让格式奖励压过正确性。否则模型可能学会输出漂亮格式但答案不对。

## 检查 parquet

准备数据后，先抽样看几行：

```bash
python - <<'PY'
import pandas as pd

path = "/Users/meisen/data/gsm8k/train.parquet"
df = pd.read_parquet(path)
print(df.columns)
print(df.iloc[0].to_dict())
PY
```

如果在 GPU 机器上使用 `~` 路径：

```bash
python - <<'PY'
import os
import pandas as pd

path = os.path.expanduser("~/data/gsm8k/train.parquet")
df = pd.read_parquet(path)
print(df.columns.tolist())
print(df.iloc[0].to_dict())
PY
```

检查清单：

- `prompt` 或 `messages` 是否是 list of dict；
- role 是否符合模型 chat template；
- `ground_truth` 是否和 reward parser 一致；
- 样本里是否混入 test 数据；
- prompt 长度是否超过 `data.max_prompt_length`；
- assistant 答案是否超过 `data.max_response_length`；
- `data_source` 是否和 reward / teacher route 匹配；
- Agentic RL 是否有 `agent_name`。

## 数据切分

不要把全部数据都拿来训练。至少保留：

- train：更新模型；
- validation：训练中快速看趋势；
- test：阶段结束后只跑少数几次；
- holdout prompts：人工抽检和 regression。

对小实验，可以先用 128 条训练样本和 64 条验证样本跑通。对正式实验，再扩大到完整数据。

## 初学者最常见错误

### SFT 用错数据格式

SFT 脚本默认 `data.messages_key=messages`。如果你喂的是 RL 数据里的 `prompt`，训练器可能找不到字段或训练目标不对。

### RL 数据没有 `reward_model.ground_truth`

没有 ground truth，规则 reward 不知道怎么判分。reward 全是 0 时，GRPO 不会学到东西。

### Agentic RL 忘记 `agent_name`

工具数据没有 `agent_name`，rollout 可能走默认单轮 agent，工具调用不会真正发生。

### prompt 和 reward 约定不一致

prompt 要求输出 `#### <answer>`，reward 却解析 `\boxed{}`，最终 reward 会大量为 0。

### 过早混合复杂数据

先用一个数据源跑通，再混 GSM8K、MATH、代码、agent。混合数据会让 debug 难度成倍增加。

<div class="checkpoint">

**本章结论**

verl 的数据核心是 parquet。SFT 看 `messages`，RL/RLVR 看 `prompt + reward_model`，Agentic RL 还需要 `agent_name` 和工具参数，偏好优化看 `prompt/chosen/rejected`。训练前先抽样读 parquet，比直接开跑更省算力。

</div>
