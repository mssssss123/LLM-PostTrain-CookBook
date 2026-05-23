# 12. 实验手册：从想法到复盘

这章给一套可执行流程。无论你做 SFT、DPO、RLVR 还是 OPD，都可以按这个手册推进。后训练不是一次性调参，而是一套实验科学：先提出假设，再构造数据和评估，最后用结果决定下一步。

## 0. 写清研究问题

不要从“我要训一下”开始。先写：

```text
问题：模型在中文数学解释题中经常跳步。
假设：加入高质量分步 SFT 数据，并用格式评估约束，可以提升解释完整性。
成功标准：人工评估中“步骤完整”胜率提升 15%，同时答案正确率不下降。
```

一个好问题包含目标能力、方法假设和成功标准。没有这三项，训练结束后你很难判断“分数变化”到底是不是你想要的进步。

配套代码：把实验问题结构化，后面才能自动生成记录。

```python
from dataclasses import dataclass


@dataclass
class ResearchQuestion:
    problem: str
    hypothesis: str
    success_metric: str
    regression_metric: str


rq = ResearchQuestion(
    problem="Qwen3-4B-Base 在 GSM8K 上格式不稳定，reward 经常解析失败。",
    hypothesis="先做少量 SFT 激活 #### 格式，再接 GRPO，可以提升可验证正确率。",
    success_metric="GSM8K validation exact match 提升 10%",
    regression_metric="通用指令遵循样本人工抽检不下降",
)
```

## 1. 建 baseline

训练前跑：

- 目标任务评估；
- 通用回归评估；
- 成本/长度统计；
- 代表性 prompt 采样。

保存 baseline 输出。后面所有比较都用它。baseline 不只是一个分数，还应该包括代表性输出和错误样本，因为很多行为变化只看平均分看不出来。

## 2. 检查数据

最低要求：

- schema 校验；
- 随机抽检；
- 去重；
- 长度分布；
- 训练/评估去重；
- renderer decode；
- mask 可视化；
- 小样本过拟合。

这一步省掉，后面调参基本都是猜。数据问题会伪装成学习率问题、batch 问题、模型能力问题，所以必须在训练前处理。

配套代码：数据 schema 检查。

```python
def validate_rlvr_row(row):
    required = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    missing = [k for k in required if k not in row]
    if missing:
        return False, f"missing fields: {missing}"
    if not isinstance(row["prompt"], list):
        return False, "prompt must be list[message]"
    if "ground_truth" not in row["reward_model"]:
        return False, "reward_model.ground_truth missing"
    return True, "ok"


def validate_rows(rows):
    errors = []
    for i, row in enumerate(rows):
        ok, msg = validate_rlvr_row(row)
        if not ok:
            errors.append((i, msg))
    return errors
```

## 3. 先跑 smoke test

Smoke test 配置：

- 小模型；
- 小数据；
- 低步数；
- 频繁保存；
- 打开详细日志；
- 人工看样本。

目标是验证管线，不是追分。smoke test 只要能证明数据能读、loss/reward 能算、checkpoint 能存、样本能看，就已经完成任务。

## 4. 正式训练

每次训练记录：

```yaml
run_name: qwen35-sft-math-explain-v1
base_model: Qwen/Qwen3.5-9B-Base
method: sft
data_version: math_explain_20260523
renderer: qwen3_5
learning_rate: 2e-4
lora_rank: 32
batch_size: 128
max_length: 8192
eval_every: 50
save_every: 50
```

并保存：

- config；
- git diff；
- metrics；
- checkpoints；
- eval outputs；
- 代表性 samples。

## 5. 训练中监控

每 30 到 60 分钟看一次：

- loss/reward 是否异常；
- KL 是否过高；
- 输出长度是否飙升；
- parse failure 是否上升；
- eval 是否同步提升；
- transcript 是否可读；
- 是否有大量 timeout。

如果出现明显异常，先暂停分析，不要盲目继续烧算力。继续训练不会自动修复 reward 错误、mask 错误或环境错误，只会让错误更贵。

配套代码：用简单规则监控异常曲线。

```python
def detect_training_anomaly(metrics):
    alerts = []
    if metrics.get("kl", 0) > 0.2:
        alerts.append("KL 过高，可能策略漂移")
    if metrics.get("response_length_mean", 0) > metrics.get("baseline_length", 10**9) * 2:
        alerts.append("输出长度翻倍，可能 length hacking")
    if metrics.get("parse_failure_rate", 0) > 0.2:
        alerts.append("格式解析失败率过高")
    if metrics.get("train_reward", 0) > 0.8 and metrics.get("eval_score", 0) < 0.3:
        alerts.append("训练 reward 高但 eval 低，疑似 reward overfit")
    return alerts
```

## 6. 评估 checkpoint

不要只评估 final。选几个 checkpoint：

- early；
- middle；
- best train metric；
- best inline eval；
- final。

跑同一套 benchmark。很多训练 final 不是最佳。

## 7. 错误分析

把错误样本分类：

```text
格式错误：18%
推理错误：31%
知识错误：12%
过度拒绝：6%
截断：9%
工具参数错误：14%
其他：10%
```

每类错误对应不同修复：

| 错误 | 修复方向 |
|---|---|
| 格式错 | SFT 数据、模板、parser |
| 推理错 | RLVR、推理数据、OPD |
| 知识错 | RAG、领域数据、事实评估 |
| 拒绝错 | 偏好数据、边界样本配比 |
| 截断 | max tokens、长度训练 |
| 工具错 | 工具 SFT、多轮 RL、协议简化 |

## 8. 做消融

消融不是论文专用。工程里也要知道到底什么有效。如果一次改了数据、学习率、rank 和 prompt，你即使看到提升，也不知道是哪一项带来的。

常见消融：

- 去掉某类数据；
- 不同学习率；
- 不同 LoRA rank；
- SFT vs SFT+DPO；
- GRPO group size；
- 有无 KL penalty；
- off-policy vs on-policy distillation；
- 有无格式 reward。

每次只改一个关键变量。

配套代码：生成一个最小 sweep 表。

```python
def make_lr_sweep(base_config, lrs):
    configs = []
    for lr in lrs:
        cfg = dict(base_config)
        cfg["ACTOR_LR"] = lr
        cfg["EXPERIMENT_NAME"] = f"{base_config['EXPERIMENT_NAME']}-lr-{lr}"
        configs.append(cfg)
    return configs


base = {"EXPERIMENT_NAME": "qwen3-4b-grpo", "ROLLOUT_N": 4, "TRAIN_BATCH_SIZE": 128}
sweep = make_lr_sweep(base, ["5e-7", "1e-6", "2e-6"])
```

## 9. 复盘模板

每个实验结束写复盘：

```markdown
# Run Summary

## 目标
提升中文数学解释完整性。

## 配置
- base: ...
- method: ...
- data: ...

## 结果
- target eval: 62 -> 71
- regression eval: 88 -> 86
- avg tokens: 520 -> 690

## 发现
- 分步解释提升明显。
- 输出变长，需要长度控制。
- 错误仍集中在多条件题。

## 下一步
- 加入长度偏好 DPO。
- 补多条件题数据。
- 调低 max_tokens 评估部署成本。
```

没有复盘的实验，后面很难复用。

## 10. 什么时候停止

停止不是失败。以下情况应该停止当前方向：

- 独立评估连续多次不涨；
- reward 涨但人工质量下降；
- 修复一个能力导致多个回归；
- 数据质量成为明显瓶颈；
- 训练成本超过预期收益；
- 方法和目标不匹配。

换方法比继续调无效超参更务实。

## 一条完整路线示例

目标：训练一个中文 LLM post-training 教学助手。

1. SFT：用高质量中文教程问答教基础概念和表达风格。
2. DPO：让模型偏向更清晰、更短、更适合小白的解释。
3. RLVR：对有标准答案的 quiz 题训练正确性。
4. OPD：用强教师模型迁移复杂推理解释。
5. Eval：概念问答、术语解释、反例识别、教学质量人工评估。
6. 部署：固定模板、限制长度、保留引用和不确定性表达。

## 本章结论

Post-training 是实验科学加工程纪律。清晰假设、可靠评估、可复现日志和诚实复盘，比追逐算法缩写更能稳定提升模型。
