# 2. 数据、模板与 Renderer

后训练里最容易被低估的部分是数据渲染。很多训练失败不是因为 SFT、DPO 或 RL 算法有问题，而是因为模型看到的 token 和你以为的不一样。对语言模型来说，JSON、parquet、messages 都只是人类组织数据的方式；真正进入模型的是 token 序列，以及告诉训练器哪些 token 要计算 loss 的 mask。

这一章讲清楚四件事：

- 聊天消息如何变成 token；
- 为什么不同模型必须使用不同模板；
- 训练 mask 如何决定哪些 token 参与 loss；
- 数据检查应该怎样做。

## 模型不直接看 JSON

我们喜欢用这种结构表达对话：

```json
[
  {"role": "system", "content": "你是一个严谨的数学助教。"},
  {"role": "user", "content": "什么是 GRPO？"},
  {"role": "assistant", "content": "GRPO 是一种组内相对优势的强化学习方法..."}
]
```

但模型真正看到的是一串 token。中间需要 chat template 把 role、分隔符、结束符、thinking 标记等全部渲染出来。你在数据里写的是 `role: user`，模型看到的可能是某种特殊标记、换行符和内容拼接后的序列。

不同模型家族模板不同。Llama、Qwen、DeepSeek、Kimi、GPT-OSS 的 system/user/assistant 标记不一样；有些模型支持 `<think>`；有些模型有 tool calling 的特殊格式；视觉模型还需要图片 token 计数和 image processor。模板错了，训练不会一定报错，但模型学到的协议会和推理时使用的协议不一致。

## Chat template 是什么

Chat template 可以理解为“模型协议适配器”。它负责三件事：

1. 把结构化消息渲染成训练/生成 token。
2. 决定哪些 token 参与训练。
3. 把模型输出 token 解析回 assistant message。

在 verl 的实战路线里，这一步主要由 Hugging Face tokenizer / processor 的 `apply_chat_template` 和训练器内部 dataset 完成。你可以用下面的脚本直接检查 Qwen3-4B-Base 的渲染结果：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", trust_remote_code=True)
messages = [
    {"role": "system", "content": "你是一个严谨的数学助教。"},
    {"role": "user", "content": "什么是 GRPO？"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
print(text)
```

不要手写模板，除非你非常确定。尤其是 reasoning 模型，经常有 thinking、tool call、多轮 observation 等特殊协议。模板错了，训练可能照常跑，loss 也可能下降，但模型学到的是损坏格式。初学者应该先相信 tokenizer 自带的 `apply_chat_template`，再通过 decode 检查确认它确实符合预期。

## 生成 prompt 和监督样本

同一段消息，在生成和训练时用途不同。生成时，模型只应该看到用户已经说过的话和“现在轮到 assistant 说话”的提示；训练时，模型还会看到完整 assistant 答案，但 loss 只应该落在答案 token 上。

生成时，我们需要构造“到 assistant 开始回答之前”的 prompt：

```python
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "解释 SFT。"},
    ],
    add_generation_prompt=True,
    tokenize=False,
)
```

SFT 时，我们需要完整对话，并让训练器只对 assistant 目标 token 计算 loss。verl 的 SFT 数据通常是：

```json
{
  "messages": [
    {"role": "user", "content": "解释 SFT。"},
    {"role": "assistant", "content": "SFT 是用示范答案训练模型..."}
  ]
}
```

训练命令里通过 `data.messages_key=messages` 告诉训练器读取哪个字段：

```bash
data.messages_key=messages
```

在多轮 SFT 中，loss mask 通常只覆盖 assistant 新生成的 token。0 表示上下文，不参与 loss；1 表示模型要学习预测它。这个选择不是细节，而是训练目标本身：mask 决定模型到底在学习回答问题，还是在学习复述上下文。

## 一条 SFT 样本怎样变成 labels

初学者最容易混淆的是 `input_ids`、`labels` 和 `loss_mask`。可以先用一个极简版本理解：`input_ids` 是模型读到的完整文本，`labels` 是希望模型预测的目标，`loss_mask` 是告诉 loss 函数哪些位置算数。

假设渲染后的文本是：

```text
<user>解释 SFT。</user><assistant>SFT 是监督微调。</assistant>
```

模型输入是完整文本，但我们只想训练 assistant 答案。于是：

- `input_ids`：完整 token 序列，包含 user 和 assistant。
- `labels`：和 `input_ids` 等长，但 user token 位置设成 `-100`。
- `loss_mask`：assistant token 是 1，其他位置是 0。

PyTorch 里的教学版构造如下：

```python
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", trust_remote_code=True)

prompt_messages = [
    {"role": "user", "content": "解释 SFT。"},
]
full_messages = [
    {"role": "user", "content": "解释 SFT。"},
    {"role": "assistant", "content": "SFT 是用示范答案训练模型，让模型学会按指令回答。"},
]

prompt_text = tokenizer.apply_chat_template(
    prompt_messages,
    add_generation_prompt=True,
    tokenize=False,
)
full_text = tokenizer.apply_chat_template(
    full_messages,
    add_generation_prompt=False,
    tokenize=False,
)

input_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids
prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids

labels = input_ids.clone()
labels[:, : prompt_ids.shape[1]] = -100
loss_mask = labels.ne(-100).long()

print("input length:", input_ids.shape[1])
print("train tokens:", loss_mask.sum().item())
```

真实训练器还要处理 padding、多轮 assistant、截断、图片/视频 token、工具返回和分布式 batch，但核心就是：**输入可以看完整上下文，loss 只落在要学习的 assistant token 上**。

## Mask 怎么选

| Mask 策略 | 适合场景 | 风险 |
|---|---|---|
| 只训练最后一条 assistant | 单轮或只关心最终答案 | 多轮过程能力学得少 |
| 训练所有 assistant | 多轮对话 SFT，训练每轮助手回复 | 中间低质量回复也会被学进去 |
| 训练完整文本 | 语言建模或特殊实验 | 容易让模型学 user/system 文本 |
| 自定义 mask | 工具轨迹、错误动作 masking | 需要严格测试，否则容易 mask 错 |

对初学者，默认从“只训练 assistant”开始。工具 observation、用户消息、系统提示通常不应该作为 assistant 目标训练。只有当你明确知道自己在做语言建模、prompt distillation 或特殊协议训练时，才应该考虑训练更大范围的 token。

## 数据格式建议

最通用的 SFT JSONL 格式：

```json
{"messages":[{"role":"user","content":"什么是 KL？"},{"role":"assistant","content":"KL 散度衡量两个分布的差异..."}]}
{"messages":[{"role":"user","content":"用一句话解释 DPO。"},{"role":"assistant","content":"DPO 是一种直接用偏好对优化语言模型的方法。"}]}
```

偏好数据建议保留 prompt、chosen、rejected，并尽量保留消息结构：

```json
{
  "prompt": [{"role": "user", "content": "帮我润色这段话。"}],
  "chosen": [{"role": "assistant", "content": "更自然、更清楚的版本..."}],
  "rejected": [{"role": "assistant", "content": "生硬或有问题的版本..."}]
}
```

RL 数据通常不是完整答案，而是环境构造参数，比如数学题的 `question/answer`、代码题的 tests、工具任务的初始状态。原因是 RL 阶段需要模型自己生成答案，然后由 reward function 或环境判断结果，而不是提前把标准 assistant 答案喂给模型模仿。

## 数据质量比数量更早成为瓶颈

对 post-training 来说，“更多数据”不一定更好。低质量数据会稳定地把模型推向错误行为。

检查数据时至少看这些维度：

- **任务一致性**：同一字段是否表达同一语义？
- **答案质量**：assistant 是否真的解决了问题？
- **格式稳定性**：是否混用 Markdown、JSON、自然语言协议？
- **长度分布**：是否有极端长样本导致截断？
- **重复率**：重复样本会放大局部风格。
- **边界样本**：困难题、需要澄清、工具失败或不可完成任务是否被标清？
- **训练/评估泄漏**：benchmark 题目是否混进训练集？

## Decode 检查

训练前必须 decode 几条样本。你要亲眼看到模型实际接收的文本。不要只看原始 JSON，因为原始 JSON 看起来正确，不代表渲染后的 token 正确。

```python
import os
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base", trust_remote_code=True)
df = pd.read_parquet(os.path.expanduser("~/data/gsm8k_sft/train.parquet"))
messages = df.iloc[0]["messages"]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=False,
    tokenize=False,
)
tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

print(text[:3000])
print("num_tokens:", len(tokens))
```

检查点：

- system/user/assistant 分隔符是否符合模型家族？
- assistant 的结束符是否存在？
- 训练权重是否只覆盖你想训练的区域？
- 超过 `max_length` 时截断发生在哪里？
- 多轮数据中，是否错误训练了工具返回或用户消息？

<div class="warning">

**静默错误**

Renderer 错误、mask 错误、EOS 错误通常不会立刻报错。训练会继续，loss 也可能下降，但模型行为会变差。这类问题只能靠 decode、单元测试和小样本过拟合提前发现。

</div>

## Thinking 模型的特殊问题

推理模型常见 `<think>...</think>`。训练时要明确：

- 是否训练 thinking 内容？
- 是否要求输出 thinking？
- 是否使用 disable-thinking renderer？
- 评估时是否剥离 thinking 再判分？

如果训练数据包含 thinking，而部署时又要求直接回答，模型会被拉向不一致目标。反过来，如果你想训练数学推理能力，完全去掉 thinking 也可能损失过程学习信号。

经验规则：

- 做直接指令遵循：优先 disable-thinking 模式。
- 做推理/RLVR：保留或显式控制 thinking 格式。
- 做评估：判分函数应明确是否忽略 thinking。

## 工具调用数据

工具调用模型不仅要输出自然语言，还要输出结构化 action。一个工具调用样本通常包含：

- system 中的工具说明；
- user 的任务；
- assistant 的 tool call；
- tool 返回结果；
- assistant 的最终回答。

多轮工具训练时，环境返回的 tool observation 通常不应该作为 assistant token 训练。否则模型可能学会“伪造工具结果”。正确做法是让模型学习什么时候发起 tool call、如何读取 observation、何时给 final answer，而不是学习生成 observation 本身。

## 推荐的数据检查脚本

最小检查脚本应该输出：

1. 原始 JSON。
2. 渲染后的文本。
3. token 长度。
4. 可训练 token 比例。
5. 截断比例。
6. 随机 20 条人工抽检。

最小版本：

```python
import os
import random
import pandas as pd
from transformers import AutoTokenizer


def inspect_sft_parquet(path, model_name, n=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    df = pd.read_parquet(os.path.expanduser(path))
    for i in random.sample(range(len(df)), min(n, len(df))):
        messages = df.iloc[i]["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        print("=" * 80)
        print("example", i, "tokens", len(token_ids))
        print(text[:3000])


inspect_sft_parquet("~/data/gsm8k_sft/train.parquet", "Qwen/Qwen3-4B-Base")
```

这个脚本只检查文本渲染。正式训练前，还应该抽样检查 labels 或 loss mask，确认 prompt 部分没有被训练、assistant 答案没有被漏掉、长样本截断没有切掉关键答案。

## 本章结论

在后训练项目里，数据渲染是第一等公民。你不是在训练 JSON，也不是在训练“你脑子里想象的对话”。你训练的是 renderer 产出的 token 序列和 mask。

下一章进入 SFT：在数据渲染正确的前提下，怎样用示范把行为写进模型。
