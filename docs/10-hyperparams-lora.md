# 10. LoRA、超参与训练稳定性

后训练的超参不需要神秘化。大多数问题可以先用一套保守起点跑通，再用曲线和样本决定调整方向。初学者最容易犯的错是还没确认数据、mask、reward 是否正确，就开始扫学习率和 batch；这样只会把基础错误包装成复杂调参问题。

这一章讲 LoRA、学习率、batch、长度、KL 和常见曲线诊断。

## LoRA 是什么

LoRA 的直觉：不直接更新原模型的大矩阵，而是在旁边加一组低秩可训练矩阵。训练后得到一个 adapter。这样做的好处是你不用改动全部权重，就能让模型在某个任务或风格上发生明显变化。

优点：

- 训练参数少；
- 成本低；
- 容易保存和切换；
- 可以合并回基座模型；
- 适合快速实验。

缺点：

- 容量有限；
- 复杂任务可能需要更高 rank；
- 多个 adapter 合并和管理要小心；
- 极端分布迁移时不如全参灵活。

## LoRA rank

rank 控制 adapter 容量。rank 越高，adapter 能表达的变化越复杂，但训练和存储成本也更高。

| rank | 适合场景 |
|---:|---|
| 8 到 16 | 简单格式、轻量风格 |
| 32 | 常规 SFT / DPO 起点 |
| 64 | 复杂对话、领域任务 |
| 128 | 推理蒸馏、复杂能力迁移 |

rank 不是越高越好。rank 高会增加成本，也可能更容易过拟合。先用 32 建 baseline，再根据任务复杂度上调。如果只是学习格式和轻量风格，rank 过高通常没有必要；如果是复杂推理蒸馏或多轮工具行为，rank 太低可能容量不够。

## LoRA 层的最小实现

LoRA 做的事可以写成：

```text
y = x W^T + scale * x A^T B^T
```

其中原始权重 `W` 冻结，只训练小矩阵 `A` 和 `B`。

教学版 PyTorch：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank=32, alpha=16):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_a = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_out + self.scale * lora_out
```

这就是为什么 `LORA_RANK` 决定容量，`LORA_ALPHA` 决定 adapter 输出的缩放。SFT、DPO、GRPO 都可以训练 LoRA；区别在于 loss 不同。

## 学习率

学习率决定每步更新幅度。它不是“越大学得越快”这么简单：学习率过高会让模型快速偏离原能力，学习率过低会让训练看起来稳定但没有行为变化。

经验：

- SFT LoRA 通常高于 RL/DPO；
- DPO 常从 `1e-5` 开始；
- RL 常从 `1e-5` 到 `4e-5`；
- OPD/蒸馏根据 KL 和任务可从 `1e-4` 试起；
- 大模型通常需要更保守的学习率。

在本教程的 verl 实战里，可以先用脚本暴露的环境变量设学习率：SFT LoRA 从 `LR=1e-4` 起步，GRPO/OPD 的 actor 从 `ACTOR_LR=1e-6` 起步。它们只是保守起点，不是免调参。

## Batch size

SFT 的 batch 可以按样本或 token 理解，但真正影响计算的是 token。长样本会让实际 token batch 变大。两个 batch 都是 64 条样本，token 数可能差几倍，显存和梯度稳定性也会差很多。

RL 中常分两层：

- `groups_per_batch`：每步多少个问题；
- `group_size`：每个问题采样多少答案。

总 rollout 数：

```text
total_rollouts = groups_per_batch * group_size
```

`group_size` 增大能降低组内 advantage 噪声，但成本上升。初期可以小一点做调试，正式训练再扩大。

## Max length 和 max tokens

常见长度参数：

- SFT `max_length`：prompt + answer 总长度；
- RL `max_tokens`：单次生成最大 token；
- 多轮 `max_trajectory_tokens`：整条轨迹最大 token。

长度太短会截断答案，长度太长会增加成本并鼓励啰嗦。推理模型尤其要小心，因为 thinking 会占大量 token。调长度时不要只看能不能跑，还要看答案是否完整、reward parser 是否能解析、平均 token 成本是否可接受。

## KL 和 beta

KL 控制模型偏离参考策略的程度。

| 场景 | 控制项 |
|---|---|
| RLHF/PPO | KL penalty |
| GRPO/RLVR | KL penalty 或训练 loss 配置 |
| DPO | `dpo_beta` |
| OPD | teacher-student KL |

如果模型输出开始变怪、格式崩、边界行为漂移，先看 KL。KL 过高时，降低学习率、提高 KL penalty、缩短训练或改初始化。

## 曲线诊断

### SFT

| 曲线 | 可能含义 |
|---|---|
| train loss 下降，eval loss 下降 | 正常学习 |
| train loss 下降，eval loss 上升 | 过拟合 |
| loss 不动 | LR 太低、mask 错、数据错 |
| loss 快速到很低 | 数据重复或任务太简单 |

### RL

| 曲线 | 可能含义 |
|---|---|
| reward 稳定上升，eval 上升 | 健康 |
| reward 上升，eval 不动 | reward 过拟合 |
| KL 快速升高 | 更新太激进 |
| entropy 快速下降 | 探索不足 |
| response length 飙升 | 模型在用长度刷 reward |
| parse failure 上升 | 格式崩 |

### DPO

| 曲线 | 可能含义 |
|---|---|
| preference loss 下降，win-rate 上升 | 健康 |
| 输出显著变长 | 长度偏差 |
| 拒绝率上升 | 边界类偏好过强或数据配比不当 |
| 通用能力下降 | beta/LR/数据混合需调整 |

## Sweep 策略

不要一次 sweep 太多维度。推荐顺序：

1. 固定数据和模型，扫学习率。
2. 固定最好 LR，扫 LoRA rank。
3. RL 再扫 group size / KL。
4. 最后扫训练步数和数据配比。

每次 sweep 都要记录：

- git commit；
- 数据版本；
- config；
- checkpoint；
- eval 结果；
- 代表性失败样本。

## 稳定训练的默认起点

| 任务 | LR | rank | batch |
|---|---:|---:|---|
| 聊天 SFT | `2e-4` | 32 | 128 |
| 领域 SFT | `1e-4` | 32 到 64 | 64 到 128 |
| DPO | `1e-5` | 32 | 256 |
| 数学 GRPO | `4e-5` | 32 | `128 x 8/16` |
| OPD 推理 | `1e-4` | 64 到 128 | 任务相关 |
| 多轮工具 RL | `1e-5` | 32 | 小 batch 起步 |

这些只是起点。最终以独立评估和样本质量为准。

## 本章结论

超参调优不是玄学，而是诊断。先用保守配置跑通，再根据 loss、reward、KL、长度、失败率和独立评估逐步调整。
