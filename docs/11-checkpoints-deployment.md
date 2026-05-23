# 11. Checkpoint、导出与部署

训练出一个 checkpoint 不是结束。你还需要能恢复、比较、导出、发布、服务化，并在上线后回滚。对后训练来说，checkpoint 是实验资产，不只是一个权重文件；如果你不知道它来自哪份数据、哪个配置、哪次评估，就无法判断它是否值得继续使用。

这一章讲 post-training 的生命周期管理。

## Checkpoint 类型

常见有两类：

| 类型 | 用途 | 包含内容 |
|---|---|---|
| training state | 恢复训练 | 权重、优化器状态、训练进度 |
| sampler weights | 推理/评估/导出 | 可用于 sampling 的权重 |

训练中保存 state，是为了断点恢复；保存 sampler，是为了评估和部署。两者不要混淆：能恢复训练的目录不一定能直接给推理服务加载，能推理的权重也不一定包含 optimizer 状态。

## 保存策略

建议：

- 每 N step 保存 sampler checkpoint；
- 关键阶段保存 training state；
- 最终 checkpoint 永久保留；
- 中间 checkpoint 设置 TTL；
- `checkpoints.jsonl` 记录元数据；
- 每个 checkpoint 绑定 config 和 eval 结果。

不要只保留最后一个。很多训练最佳点出现在中间，后面可能过拟合或能力回退。尤其是 RL 和偏好优化，final checkpoint 经常不是最好的 checkpoint。

配套代码：训练中保存 checkpoint 元数据。

```python
import json
from pathlib import Path
from datetime import datetime


def append_checkpoint_record(run_dir, step, checkpoint_path, metrics):
    record = {
        "step": step,
        "checkpoint_path": checkpoint_path,
        "metrics": metrics,
        "created_at": datetime.utcnow().isoformat(),
    }
    path = Path(run_dir) / "checkpoints.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

这样你后面能回答“为什么选这个 step 部署”，而不是只凭印象。

## 恢复训练

恢复训练需要：

- checkpoint path；
- 原始 config；
- 数据版本；
- renderer；
- optimizer state；
- 当前 step/epoch；
- 日志目录处理策略。

如果只加载 sampler weights，通常只能继续从权重初始化训练，无法完整恢复 optimizer 动量。

配套代码：从记录里选验证分最高的 checkpoint。

```python
def select_best_checkpoint(records_path, metric_name):
    best = None
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            score = row["metrics"].get(metric_name)
            if score is None:
                continue
            if best is None or score > best["score"]:
                best = {"score": score, "checkpoint_path": row["checkpoint_path"], "step": row["step"]}
    return best
```

## 日志目录

一个好的训练目录至少包含：

```text
run/
  config.json
  metrics.jsonl
  checkpoints.jsonl
  code.diff
  iteration_000020/
    train.html
    train_logtree.json
    eval_gsm8k.html
```

`metrics.jsonl` 看曲线，`train_logtree.json` 看 rollout，`code.diff` 保证结果能追溯。一个训练目录应该能回答四个问题：用什么数据训的，怎么训的，训到哪一步，为什么选择这个 checkpoint。

## 用 verl 导出 Hugging Face 模型

verl 的 FSDP/Megatron checkpoint 不是直接可部署的 Hugging Face 目录。训练后通常要用 `verl.model_merger` 转换。

GRPO 示例的 checkpoint 默认在：

```text
checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_*/actor
```

导出 FSDP checkpoint：

```bash
cd verl-main
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/llm-posttrain-cookbook/qwen3-4b-base-gsm8k-grpo/global_step_20/actor \
  --target_dir checkpoints/llm-posttrain-cookbook/qwen3-4b-base-gsm8k-grpo/global_step_20/actor/huggingface
```

把 `global_step_20` 换成你验证集最好的 step。不要默认最后一步就是最好模型。

LoRA 训练也要区分两件事：

1. 训练 checkpoint：用于恢复训练和继续实验。
2. 推理模型：用于 vLLM/SGLang/Hugging Face Transformers 加载。

如果训练脚本启用了 LoRA merge，例如 `actor_rollout_ref.model.lora.merge=True`，rollout 阶段会把 adapter 合入 base 权重给推理引擎使用。部署前仍然要确认最终导出的目录包含完整 tokenizer、config 和权重文件。

配套代码：导出后用 Transformers 做一次 smoke test。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


def smoke_test_hf_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)

    messages = [{"role": "user", "content": "用一句话解释 SFT。"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.0)
    return tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
```

这个 smoke test 不代表模型好，只代表导出的 tokenizer、template、权重能正常加载和生成。

## 发布到 HuggingFace

发布前准备 model card。model card 不是形式文档，而是告诉别人这个模型从哪里来、适合做什么、不适合做什么、应该用什么模板调用。

- base model；
- training method；
- dataset；
- intended use；
- limitations；
- evaluation；
- license；
- boundary / risk notes；
- prompt template；
- adapter 或 full model 说明。

不要发布一个没有模板说明的模型。用户不知道该用哪个 chat template，效果会严重偏差。

## 部署前检查

| 检查项 | 为什么重要 |
|---|---|
| renderer 与推理 template 一致 | 否则训练/推理格式不一致 |
| EOS/stop 设置 | 防止无限输出或截断 |
| max tokens | 控制成本和完整性 |
| temperature/top_p | 影响评估与用户体验 |
| boundary/risk eval | 防止新模型边界行为漂移 |
| latency/cost | LoRA 和全量模型服务成本不同 |
| rollback path | 上线异常可快速回退 |

## A/B 测试

上线模型不要只看离线 benchmark。A/B 测试关注真实用户任务是否完成、成本是否可接受、失败是否可回滚。

- 用户满意度；
- 任务完成率；
- 追问率；
- 拒绝率；
- 平均 token；
- 延迟；
- 人工升级率；
- 边界和风险事件。

训练指标、离线 benchmark、线上指标三者共同决定模型是否可用。

配套代码：最小 A/B 记录。

```python
def log_ab_event(path, user_id, variant, prompt, response, metrics):
    event = {
        "user_id": user_id,
        "variant": variant,  # "baseline" or "new_model"
        "prompt": prompt,
        "response": response,
        "metrics": metrics,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
```

上线指标必须能按 `variant` 聚合，否则 A/B 只是“两个模型都上线了”，不是实验。

## 版本命名

推荐命名包含：

```text
<base>-<method>-<task>-<data_version>-<date>-<short_hash>
```

例如：

```text
qwen35-9b-sft-dpo-customer-v3-20260523-a1b2c3
```

模型版本应该能反查到代码、数据、config、checkpoint 和评估。

配套代码：生成可追溯版本名。

```python
def make_model_version(base, method, task, data_version, date, git_sha):
    return f"{base}-{method}-{task}-{data_version}-{date}-{git_sha[:7]}"


version = make_model_version(
    base="qwen3-4b-base",
    method="sft-grpo",
    task="gsm8k",
    data_version="v1",
    date="20260523",
    git_sha="a1b2c3d4e5",
)
```

## 本章结论

Checkpoint 是实验资产，不只是中间文件。没有可恢复、可比较、可导出、可回滚的训练生命周期，post-training 很难进入生产。
